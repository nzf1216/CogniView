from config import RETINA_DATA_DIR

dataset = RetinaDataset(
    csv_file=RETINA_DATA_DIR / "train.csv",
    image_dir=RETINA_DATA_DIR / "train_images",
    transform=get_train_transforms()
)

start = time.time()

image, label = dataset[0]

end = time.time()

print("Time:", end - start)
print(image.shape)
print(label)