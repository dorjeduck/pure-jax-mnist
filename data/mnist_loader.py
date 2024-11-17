import torch
from torchvision import datasets, transforms
import numpy as np

# Set up a transformation for normalization
normalize_data = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_mnist_data(batch_size=128):
    """
    Loads and preprocesses the MNIST dataset using PyTorch DataLoader.
    Returns training and test data as NumPy arrays for compatibility with JAX.
    
    Parameters:
    - batch_size (int): Number of samples per batch for loading

    Returns:
    - train_images (np.ndarray): Training images as flattened NumPy arrays
    - train_labels (np.ndarray): One-hot encoded training labels
    - test_images (np.ndarray): Test images as flattened NumPy arrays
    - test_labels (np.ndarray): One-hot encoded test labels
    """
    # Load MNIST dataset using torchvision
    train_dataset = datasets.MNIST(root='MNIST', train=True, download=True, transform=normalize_data)
    test_dataset = datasets.MNIST(root='MNIST', train=False, download=True, transform=normalize_data)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Convert data to NumPy arrays
    train_images, train_labels = pytorch_to_numpy(train_loader)
    test_images, test_labels = pytorch_to_numpy(test_loader)
    
    return train_images, train_labels, test_images, test_labels

def pytorch_to_numpy(data_loader):
    """
    Converts PyTorch DataLoader batches to NumPy arrays for compatibility with JAX.
    
    Parameters:
    - data_loader (torch.utils.data.DataLoader): PyTorch DataLoader instance

    Returns:
    - images (np.ndarray): Images as flattened NumPy arrays
    - labels (np.ndarray): Labels as one-hot encoded NumPy arrays
    """
    images, labels = [], []
    for x_batch, y_batch in data_loader:
        images.append(x_batch.view(-1, 28*28).numpy())  # Flatten the images to 784-dimensional
        labels.append(y_batch.numpy())
    
    # Stack and concatenate all batches
    images = np.vstack(images)
    labels = np.concatenate(labels)
    
    # One-hot encode labels for JAX compatibility
    labels = np.eye(10)[labels]  # 10 classes for MNIST
    
    return images, labels