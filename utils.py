import os
from PIL import Image
import torch
from torchvision import transforms

def load_image(image_path):
    """
    Loads an image from disk and converts it to RGB.
    Args:
        image_path (str): Path to the image file.
    Returns:
        PIL.Image.Image: Loaded RGB image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    image = Image.open(image_path).convert('RGB')
    return image

def get_image_transform(size=224):
    """
    Returns a torchvision transform pipeline for image preprocessing.
    Args:
        size (int): Resize dimension (size x size).
    Returns:
        torchvision.transforms.Compose: Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def preprocess_image(image_path, transform=None):
    """
    Loads and preprocesses a single image.
    Args:
        image_path (str): Path to the image.
        transform (torchvision.transforms.Compose, optional): Transform pipeline. If None, uses default.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    if transform is None:
        transform = get_image_transform()
    image = load_image(image_path)
    image_tensor = transform(image)
    return image_tensor

def batch_preprocess_images(image_paths, transform=None, device='cpu'):
    """
    Preprocesses a batch of images into a single tensor batch.
    Args:
        image_paths (list of str): List of image file paths.
        transform (torchvision.transforms.Compose, optional): Transform pipeline.
        device (str): Device to place the tensor on.
    Returns:
        torch.Tensor: Batch of preprocessed images.
    """
    if transform is None:
        transform = get_image_transform()
    images = [preprocess_image(path, transform) for path in image_paths]
    batch_tensor = torch.stack(images).to(device)
    return batch_tensor

def safe_mkdir(directory):
    """
    Safely create a directory if it does not exist.
    Args:
        directory (str): Path to directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

