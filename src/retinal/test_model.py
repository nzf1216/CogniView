import torch

from model import RetinaClassifier

model = RetinaClassifier()

dummy = torch.randn(
    4,
    3,
    224,
    224
)

output = model(dummy)

print("Output Shape:", output.shape)