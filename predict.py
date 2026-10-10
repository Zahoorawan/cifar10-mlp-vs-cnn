
import torch
from torch import nn
from PIL import Image
from torchvision import transforms

# 1. Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. CIFAR-10 classes
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 3. Recreate the same CNN you trained
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# 4. Load the trained model
model = CNN().to(device)

model.load_state_dict(
    torch.load(
        "cifar10_cnn.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

# 5. Preprocess your picture like the training images
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

# 6. Enter your image path here
image_path = "frog3.jpg"



image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# 7. Predict
with torch.no_grad():
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)

    predicted_index = probabilities.argmax(dim=1).item()

print("Predicted class:", classes[predicted_index])
print(f"Confidence: {probabilities[0, predicted_index].item() * 100:.2f}%")

