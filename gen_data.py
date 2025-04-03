import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from data_v1 import save_class_samples, save_all_classes
from dir_names import *

def generate_cifar_dataset(save_path: str, train: bool = True, verbose: bool = True):
    """Generate CIFAR dataset and save as numpy arrays."""
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    plots_path = os.path.join(save_path, 'plots')
    os.makedirs(plots_path, exist_ok=True)

    # Load CIFAR
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset_path = os.path.join(save_path, 'data')
    test_path = os.path.join(save_path, 'test')
    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        dataset = torchvision.datasets.CIFAR10(
            root=dataset_path,
            train=train,
            download=True,
            transform=transform
        )
        # Load test data
        if os.path.exists(test_path):
            print(f"Test data already exists at {test_path}")
            testset = torchvision.datasets.CIFAR10(root=test_path, train=False, download=False, transform=transform)
        else:
            print(f"Downloading test data to {test_path}")
            testset = torchvision.datasets.CIFAR10(root=test_path,
                                                train=False, 
                                                download=True,
                                                transform=transform)
        X_test = testset.data
        y_test = np.array(testset.targets)
    else:
        print(f"Downloading dataset to {dataset_path}")
        dataset = torchvision.datasets.CIFAR10(
            root=dataset_path,
            train=train,
            download=True,
            transform=transform
        )
        # Load test data
        testset = torchvision.datasets.CIFAR10(root=test_path,
                                            train=False, 
                                            download=True,
                                            transform=transform)
        X_test = testset.data
        y_test = np.array(testset.targets)
    
    print(f"Train data shape: {dataset.data.shape}")
    print(f"Test data shape: {X_test.shape}")
    # Convert to numpy arrays
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    X, Y = next(iter(data_loader))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset))
    X_test, y_test = next(iter(test_loader))

    X = np.concatenate((X, X_test), axis=0)
    Y = np.concatenate((Y, y_test), axis=0)
    
    # Save data
    if len(X.shape) == 5:
        X = X.squeeze(1)

    # Convert to numpy with memory cleanup
    try:
        # X = X.numpy()  # Ensure data is on CPU before conversion
        # Y = Y.numpy()
        
        print(f"Memory usage before saving: {X.nbytes / 1e9:.2f} GB")  # Debug print
        
        # Save with explicit flush to disk
        np.save(os.path.join(save_path, 'X.npy'), X, allow_pickle=False)
        np.save(os.path.join(save_path, 'Y.npy'), Y, allow_pickle=False)
        
        # Verify files were saved
        x_path = os.path.join(save_path, 'X.npy')
        y_path = os.path.join(save_path, 'Y.npy')
        
        if os.path.exists(x_path) and os.path.exists(y_path):
            print(f"Successfully saved files:")
            print(f"X.npy: {os.path.getsize(x_path) / 1e9:.2f} GB")
            print(f"Y.npy: {os.path.getsize(y_path) / 1e9:.2f} GB")
        else:
            print("Error: Files were not saved successfully")
            
    except Exception as e:
        print(f"Error saving arrays: {str(e)}")
        raise
    
    if verbose:
        print(f"Saved X shape: {X.shape}")
        print(f"Saved Y shape: {Y.shape}")

    # PRINTING first 10 images and labels
    print(f'Lables: {Y[:10]}')
        
    # Generate and save visualizations
    save_all_classes(dataset, plots_path)
    for class_idx in range(10):
        save_class_samples(dataset, class_idx, plots_path)

if __name__ == "__main__":
    save_path = f'{BASE_DIR}/datasets/cifar10'
    
    generate_cifar_dataset(
        save_path=save_path,
        train=True,
        verbose=True
    )