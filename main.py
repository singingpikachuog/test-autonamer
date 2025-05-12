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
MODEL_PATH = "pokemon_cnn.pt"

os.makedirs(SAVE_PATH, exist_ok=True)

# Define the CNN class that matches the original model structure
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def generate_images():
    """Generate augmented dataset from source images"""
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
        self.transform = T.Compose([
            T.Resize((224, 224)), 
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # First, get the label map from the dataset
        try:
            dataset = ImageFolder(folder, transform=self.transform)
            self.label_map = {v: k for k, v in dataset.class_to_idx.items()}
            print(f"Found {len(self.label_map)} classes in the dataset")
        except Exception as e:
            print(f"Error reading dataset: {e}")
            self.label_map = {}
        
        # Now try to load the model or create a new one
        if os.path.exists(model_path):
            try:
                print(f"Attempting to load model from {model_path}...")
                # Register the CNN class as safe before loading
                torch.serialization.add_safe_globals([CNN])
                self.model = torch.load(model_path, map_location=self.device, weights_only=False)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model safely: {e}")
                try:
                    print("Attempting to load model with weights_only=False (less secure but might work)...")
                    self.model = torch.load(model_path, map_location=self.device, weights_only=False)
                    print("Model loaded successfully!")
                except Exception as e2:
                    print(f"Error loading model unsafely: {e2}")
                    print("Will train a new model instead.")
                    self.model, self.label_map = self._train(folder)
        else:
            print(f"Model file {model_path} not found, training a new model...")
            self.model, self.label_map = self._train(folder)

        # Make sure model is in eval mode for inference
        self.model.eval()

    def _train(self, folder):
        """Train a new model on the dataset"""
        print("Preparing to train new model...")
        dataset = ImageFolder(folder, transform=self.transform)
        self.label_map = {v: k for k, v in dataset.class_to_idx.items()}
        
        # Check if we have enough data to train
        if len(dataset) < 10:
            raise ValueError(f"Not enough training data: found only {len(dataset)} images")
            
        # Create data loader
        loader = DataLoader(dataset, batch_size=32, shuffle=True, 
                          num_workers=0, pin_memory=True)
        
        # Create the model - we'll use ResNet18 as it's more reliable
        model = models.resnet18(weights='IMAGENET1K_V1').to(self.device)
        model.fc = nn.Linear(model.fc.in_features, len(self.label_map)).to(self.device)
        
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        
        # Use torch.amp for mixed precision training if available
        if torch.cuda.is_available():
            # Use the new recommended way for torch.amp
            if hasattr(torch.amp, 'GradScaler'):
                scaler = torch.amp.GradScaler('cuda')
            else:
                scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
            
        print(f"Starting training with {len(dataset)} images across {len(self.label_map)} classes")
        
        total_epochs = 5  # Reduced for faster training
        for epoch in range(total_epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # Training loop
            for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}"):
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                
                # Use mixed precision for faster training if available
                if scaler:
                    with torch.cuda.amp.autocast():
                        logits = model(x)
                        loss = loss_fn(logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    logits = model(x)
                    loss = loss_fn(logits, y)
                    loss.backward()
                    opt.step()
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            
            # Print epoch metrics
            accuracy = 100 * correct / total if total > 0 else 0
            print(f"Epoch {epoch+1}/{total_epochs} - Loss: {total_loss/len(loader):.4f} - Accuracy: {accuracy:.2f}%")

        # Save the model
        print(f"Training complete. Saving model to {self.model_path}")
        torch.save(model, self.model_path)
            
        return model, self.label_map

    def predict_url(self, url):
        """Predict Pokemon from image URL"""
        try:
            print(f"Fetching image from URL...")
            r = requests.get(url, timeout=10)
            img_arr = np.frombuffer(r.content, np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            
            if img is None:
                print("Failed to decode image")
                return None, 0.0
                
            print(f"Image loaded: {img.shape}")
            
            # Convert BGR to RGB and process image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            inputs = self.transform(pil_img).unsqueeze(0).to(self.device)

            # Get prediction
            with torch.no_grad():
                logits = self.model(inputs)
                probs = torch.softmax(logits, dim=1)
                
                # Get top 3 predictions
                top_k = min(3, len(self.label_map))
                values, indices = torch.topk(probs, top_k)
                
                # Get the best prediction
                top_idx = indices[0][0].item()
                top_prob = values[0][0].item()
                
                # Print top predictions for debugging
                print("Top predictions:")
                for i in range(top_k):
                    idx = indices[0][i].item()
                    prob = values[0][i].item()
                    print(f"  {self.label_map[idx]}: {prob*100:.2f}%")
                
                return self.label_map[top_idx], round(top_prob * 100, 2)
                
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, 0.0

if __name__ == "__main__":
    print("Pokemon CNN Classifier")
    print("=" * 40)
    
    # Check if data directory exists
    if not os.path.exists(SOURCE_IMAGE_PATH):
        print(f"Source image directory not found: {SOURCE_IMAGE_PATH}")
        print("Please ensure the data directory structure is correct.")
        exit(1)
    
    # Generate augmented dataset if needed
    if not os.path.exists(SAVE_PATH) or len(os.listdir(SAVE_PATH)) == 0:
        print("No processed images found. Generating dataset...")
        generate_images()
    else:
        print(f"Found existing processed images in {SAVE_PATH}")
    
    print("\nInitializing model...")
    try:
        net = PokeNet()
        print("\nModel ready! Enter image URLs to classify Pokemon.")
        
        while True:
            url = input("\nPaste image URL (or 'q' to quit): ").strip()
            if url.lower() == 'q': 
                print("Goodbye!")
                break
                
            print("Processing...")
            name, acc = net.predict_url(url)
            
            if name:
                print(f"🎯 Result: {name} ({acc}% confidence)")
            else:
                print("❌ Could not identify Pokemon.")
                
    except KeyboardInterrupt:
        print("\nExiting program...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()