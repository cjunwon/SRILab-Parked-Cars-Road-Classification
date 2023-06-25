import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim

# Load the pretrained MobileNetV2 model
model = models.mobilenet_v2(pretrained=True)

# Modify the last fully connected layer
num_classes = 2  # Paved and Unpaved
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)

# Helper functions used for printing
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def print_model_details(model):
    print(model)
    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")


# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Define transforms to be applied to the training images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Prepare the training dataset (Assuming you have your dataset stored in a folder called 'train_data')
train_dataset = torchvision.datasets.ImageFolder('../data/rf_data/train/', transform=transform)

# Create the train_loader
batch_size = 32  # Number of images per batch
shuffle = True  # Shuffle the training data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)


# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.requires_grad_()  # Set requires_grad to True for images


        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(running_loss)

    # Calculate average training loss for the epoch
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")