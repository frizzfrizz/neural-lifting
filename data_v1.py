import os
import numpy as np
import matplotlib.pyplot as plt
from dir_names import *
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid



# Helper function to show images
def show_images_grid(imgs_, num_images=25, save_path=None):
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            ax.imshow(imgs_[ax_i], cmap='Greys_r',  interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
        
    plt.tight_layout()
    if save_path is not None:
        if not os.path.isabs(save_path):
                save_path = os.path.join(BASE_DIR, save_path)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
# CIFAR 10 DATASET

def get_cifar10(root='./data', train=True):
    """Download and load CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root=root, 
        train=train,
        download=True, 
        transform=transform
    )
    return dataset

def tensor_to_img(tensor):
    """Convert normalized tensor to displayable image."""
    img = tensor.numpy().transpose((1, 2, 0))
    img = img * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    return np.clip(img, 0, 1)

# def show_class_samples(dataset, class_idx, samples_per_class=10, figsize=(15,15)):
#     """Display grid of images from specified class."""
#     class_samples = []
#     class_count = 0
    
#     # CIFAR-10 class names
#     classes = ('plane', 'car', 'bird', 'cat', 'deer',
#               'dog', 'frog', 'horse', 'ship', 'truck')
    
#     for img, label in dataset:
#         if label == class_idx and class_count < samples_per_class:
#             class_samples.append(img)
#             class_count += 1
#         if class_count >= samples_per_class:
#             break
    
#     # Create grid
#     grid = make_grid(class_samples, nrow=5, padding=2)
#     plt.figure(figsize=figsize)
#     plt.imshow(tensor_to_img(grid))
#     plt.title(f'Class: {classes[class_idx]}')
#     plt.axis('off')
#     plt.show()

def show_all_classes(dataset, samples_per_class=10, figsize=(15,15)):
    """Display samples from all classes in a grid."""
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')
    
    class_samples = {i: [] for i in range(10)}
    
    for img, label in dataset:
        if len(class_samples[label]) < samples_per_class:
            class_samples[label].append(img)
    
    # Combine all samples
    all_samples = []
    for class_idx in range(10):
        all_samples.extend(class_samples[class_idx])
    
    # Create grid
    grid = make_grid(all_samples, nrow=samples_per_class, padding=2)
    plt.figure(figsize=figsize)
    plt.imshow(tensor_to_img(grid))
    plt.title('CIFAR-10 Classes')
    plt.axis('off')
    plt.show()

def save_class_samples(dataset, class_idx, save_path, samples_per_class=10, figsize=(15,15)):
    """Save grid of images from specified class."""
    class_samples = []
    class_count = 0
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    for img, label in dataset:
        if label == class_idx and class_count < samples_per_class:
            class_samples.append(img)
            class_count += 1
        if class_count >= samples_per_class:
            break
    
    grid = make_grid(class_samples, nrow=5, padding=2)
    plt.figure(figsize=figsize)
    plt.imshow(tensor_to_img(grid))
    plt.title(f'Class: {classes[class_idx]}')
    plt.axis('off')
    
    # Save figure
    save_file = os.path.join(save_path, f'cifar10_class_{classes[class_idx]}.png')
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.close()

def save_all_classes(dataset, save_path, samples_per_class=10, figsize=(15,15)):
    """Save samples from all classes in a grid."""
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    class_samples = {i: [] for i in range(10)}
    
    for img, label in dataset:
        if len(class_samples[label]) < samples_per_class:
            class_samples[label].append(img)
    
    all_samples = []
    for class_idx in range(10):
        all_samples.extend(class_samples[class_idx])
    
    grid = make_grid(all_samples, nrow=samples_per_class, padding=2)
    plt.figure(figsize=figsize)
    plt.imshow(tensor_to_img(grid))
    plt.title('CIFAR-10 Classes')
    plt.axis('off')
    
    # Save figure
    save_file = os.path.join(save_path, 'cifar10_all_classes.png')
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.close()