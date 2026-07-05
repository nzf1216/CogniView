import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import IMAGE_SIZE


def get_train_transforms():
    return A.Compose([
        A.Resize(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        A.HorizontalFlip(
            p=0.5
        ),

        A.RandomBrightnessContrast(
            p=0.2
        ),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2()
    ])


def get_valid_transforms():
    return A.Compose([
        A.Resize(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2()
    ])