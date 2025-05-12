import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import requests
from tqdm import tqdm

SOURCE_IMAGE_PATH = "data/commands/pokemon/pokemon_images"
SAVE_PATH = "data/commands/pokemon/images"
MODEL_PATH = "model/pokemon_cnn.onnx"

os.makedirs(SAVE_PATH, exist_ok=True)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256 * 14 * 14, 512), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def generate_images():
    print("Generating augmented dataset...")
    count = 0
    for label in os.listdir(SOURCE_IMAGE_PATH):
        label_path = os.path.join(SOURCE_IMAGE_PATH, label)
        if not os.path.isdir(label_path): continue
        output_dir = os.path.join(SAVE_PATH, label)
        os.makedirs(output_dir, exist_ok=True)
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.resize(img, (128, 128))
            base = os.path.splitext(img_file)[0]
            cv2.imwrite(os.path.join(output_dir, f"{base}_0.png"), img)
            cv2.imwrite(os.path.join(output_dir, f"{base}_1.png"), cv2.flip(img, 1))
            count += 2
    print(f"Generated {count} augmented images")

class PokeNet:
    def __init__(self, folder=SAVE_PATH, model_path=MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model_path = model_path
        self.transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        try:
            dataset = ImageFolder(folder, transform=self.transform)
            self.label_map = {v: k for k, v in dataset.class_to_idx.items()}
            print(f"Found {len(self.label_map)} classes in the dataset")
        except Exception as e:
            print(f"Error reading dataset: {e}")
            self.label_map = {}
        
        if os.path.exists(model_path):
            try:
                print(f"Attempting to load model from {model_path}...")
                self.model = torch.load(model_path, map_location=self.device)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = self.make_onnx(folder)
        else:
            print(f"Model file {model_path} not found, generating ONNX model...")
            self.model = self.make_onnx(folder)

        self.model.eval()

    def make_onnx(self, folder):
        print("Generating ONNX model...")
        dataset = ImageFolder(folder, transform=self.transform)
        self.label_map = {v: k for k, v in dataset.class_to_idx.items()}
        model = CNN(len(self.label_map)).to(self.device)
        model.eval()

        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

        for epoch in range(5):
            model.train()
            for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/5"):
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()

        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        torch.onnx.export(model, dummy_input, self.model_path)
        print(f"ONNX model saved to {self.model_path}")
        return model

    def predict_url(self, url):
        try:
            print(f"Fetching image from URL...")
            r = requests.get(url, timeout=10)
            img_arr = np.frombuffer(r.content, np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is None: return None, 0.0
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            inputs = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(inputs)
                probs = torch.softmax(logits, dim=1)
                top_k = min(3, len(self.label_map))
                values, indices = torch.topk(probs, top_k)
                top_idx = indices[0][0].item()
                top_prob = values[0][0].item()

                return self.label_map[top_idx], round(top_prob * 100, 2)
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, 0.0

if __name__ == "__main__":
    print("Pokemon CNN Classifier")
    print("=" * 40)
    
    if not os.path.exists(SOURCE_IMAGE_PATH):
        print(f"Source image directory not found: {SOURCE_IMAGE_PATH}")
        exit(1)
    
    if not os.path.exists(SAVE_PATH) or len(os.listdir(SAVE_PATH)) == 0:
        print("No processed images found. Generating dataset...")
        generate_images()
    
    print("\nInitializing model...")
    try:
        net = PokeNet()
        print("\nModel ready! Enter image URLs to classify Pokemon.")
        
        while True:
            url = input("\nPaste image URL (or 'q' to quit): ").strip()
            if url.lower() == 'q': break
            print("Processing...")
            name, acc = net.predict_url(url)
            if name: print(f"🎯 Result: {name} ({acc}% confidence)")
            else: print("❌ Could not identify Pokemon.")
    except KeyboardInterrupt:
        print("\nExiting program...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
