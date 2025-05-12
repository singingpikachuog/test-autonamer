import os
from PIL import Image

SRC = "data/commands/pokemon/pokemon_images"
DST = "data/commands/pokemon/images"

def generate_augmented_images():
    os.makedirs(DST, exist_ok=True)
    for name in os.listdir(SRC):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        base = os.path.splitext(name)[0]
        dst_path = os.path.join(DST, base)
        os.makedirs(dst_path, exist_ok=True)
        try:
            img_path = os.path.join(SRC, name)
            img = Image.open(img_path).convert("RGBA")
            img.save(os.path.join(dst_path, "0.png"))  # original
            img.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(dst_path, "1.png"))  # flipped
        except Exception as e:
            print(f"Error processing {name}: {e}")

if __name__ == "__main__":
    print("Generating training data...")
    generate_augmented_images()
    print("Done.")
