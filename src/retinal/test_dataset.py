from dataset import RetinaDataset
from transforms import get_train_transforms

dataset = RetinaDataset(
    csv_file="../../data/retina/train.csv",
    image_dir="../../data/retina/train_images",
    transform=get_train_transforms()
)

print("Dataset Size:", len(dataset))

image, label = dataset[0]

print("Image Shape:", image.shape)

print("Label:", label)