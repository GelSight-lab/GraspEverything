import torch
import cv2
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_VAL_SPLIT = 0.2

INDENTER_SHAPE = 'ALL'
assert INDENTER_SHAPE in ['ALL', 'SPHERE', 'SQUARE', 'CONE']

# Import ResNet18 model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

class ImageRegressionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Path to the dataset folder containing subfolders.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Iterate over subfolders
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                image_path = os.path.join(subdir_path, "diff.jpg")
                label_path = os.path.join(subdir_path, "F.npy")
                if os.path.exists(image_path) and os.path.exists(label_path):
                    self.samples.append((image_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        # Load label
        label = np.load(label_path).astype(np.float32)
        label = label.item()  # Convert to scalar float

        # Convert to PyTorch tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # HWC -> CHW
        label = torch.tensor(label, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(data_dir, batch_size=32, image_size=224, num_workers=0):
    """Creates DataLoader for training and validation."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if INDENTER_SHAPE != 'ALL':
        dataset = ImageRegressionDataset(f"{data_dir}/{INDENTER_SHAPE}", transform=transform)
    else:
        dataset = ImageRegressionDataset(f"{data_dir}/SPHERE", transform=transform) + \
            ImageRegressionDataset(f"{data_dir}/SQUARE", transform=transform) + \
            ImageRegressionDataset(f"{data_dir}/CONE", transform=transform)

    val_size = int(len(dataset)*TRAIN_VAL_SPLIT)

    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(data_dir, batch_size)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        
        train_total_sq_err = 0
        train_total_images = 0
        val_total_sq_err = 0
        val_total_images = 0

        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Step
            outputs = model(images)
            outputs = torch.nn.functional.relu(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_total_sq_err += torch.sum(torch.square(outputs - labels))
            train_total_images += len(outputs)

        model.eval()

        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            val_total_sq_err += torch.sum(torch.square(outputs - labels))
            val_total_images += len(outputs)
            
        # Compute RMSE
        train_rmse = torch.sqrt(train_total_sq_err / train_total_images)
        val_rmse = torch.sqrt(val_total_sq_err / val_total_images)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train RMSE: {train_rmse:.3f}, Val. RMSE: {val_rmse:.3f}")

    return model

# Example usage
data_dir = "./software/data"
model = train_model(data_dir)
