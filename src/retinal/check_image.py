from PIL import Image
import time

path = r"C:\CogniView\data\retina\train_images\0a4e1a29ffff.png"

start = time.time()

img = Image.open(path).convert("RGB")

print("Size:", img.size)
print("Time:", time.time() - start)