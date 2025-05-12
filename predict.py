import onnx
import onnxruntime  
import numpy as np
import cv2
from PIL import Image
import requests


MODEL_PATH = "model/pokemon_cnn.onnx"

class PokeNet:
    def __init__(self, model_path=MODEL_PATH):
        # Load ONNX model
        self.model = onnx.load(model_path)
        self.input_name = self.model.graph.input[0].name
        self.output_name = self.model.graph.output[0].name
        print("Model loaded successfully!")

    def transform_image(self, img):
        # Resize and normalize image
        img = cv2.resize(img, (224, 224))  # Resize image to 224x224
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = img.astype(np.float32) / 255.0  # Normalize image to [0, 1]
        img = np.transpose(img, (2, 0, 1))  # Convert to CHW format
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    def predict_url(self, url):
        try:
            print(f"Fetching image from URL... {url}")
            r = requests.get(url, timeout=10)
            img_arr = np.frombuffer(r.content, np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is None: return None, 0.0

            # Preprocess image
            img = self.transform_image(img)

            # Run inference using ONNX
            session = onnxruntime.InferenceSession(self.model)
            inputs = {self.input_name: img}
            outputs = session.run([self.output_name], inputs)
            logits = outputs[0]

            # Get top class prediction
            top_idx = np.argmax(logits)
            top_prob = np.max(logits)

            return f"Pokemon {top_idx}", round(top_prob * 100, 2)

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, 0.0

if __name__ == "__main__":
    print("Pokemon CNN Classifier")
    print("=" * 40)

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
