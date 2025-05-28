# visualisers_v1.py
import traceback
import numpy as np
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dir_names import *

def visualise_dataset(X, y, title, save_path=None, show=False, caption=None, log_scale=False):
    """
    Visualize a dataset of multidimensional tensors.

    Parameters:
    X (numpy.ndarray): The input dataset, a 1D array.
    y (numpy.ndarray): The target dataset, a 1D array.
    """
    if log_scale:
        y = np.log(y)
    # check for numpy
    if isinstance(X, torch.Tensor):
        X = X.detach().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().numpy()
    fig, ax = plt.subplots()
    ax.scatter(X, y)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    if caption is not None:
        ax.text(0.5, 0.5, caption, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    if save_path is not None:
        # the save path should include the BASE_DIR, if not, throw up error
        if not save_path.startswith(BASE_DIR):
            raise ValueError("Save path must be within the project directory.")
        plt.savefig(save_path)
    # if show:
    #     plt.show()
    plt.close('all')

def plot_loss(metrics: dict, title: str, save_path: str=None, show: bool=False, log_scale: bool=True):
    """
    Function to plot the losses of a model
    Args:
        metrics: dict: dictionary containing the losses
        title: str: title of the plot
        save_path: str: path to save the plot
        show: bool: whether to show the plot or not
    """
    if log_scale:
        plt.yscale('log')
        metrics = {k: np.log(v) for k, v in metrics.items()}
    # two separate plots side by side for train and test losses
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(metrics['train_loss'], label='train loss')
    ax[0].set_title('Train Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(metrics['test_loss'], label='test loss')
    ax[1].set_title('Test Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    fig.suptitle(title)
    if save_path is not None:
        if not save_path.startswith(BASE_DIR):
            raise ValueError("Save path must be within the project directory.")
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close('all')

def combine_images(image_paths, grid_size, save_path, image_size=None):
    """
    Combine multiple images into a single image in a specified grid formation.

    Parameters:
    image_paths (list or str): List of image paths or directory containing images.
    grid_size (tuple): Tuple specifying the grid formation (n, m).
    save_path (str): Path to save the combined image.
    image_size (tuple, optional): Size to which each image should be resized (width, height).
    """
    if isinstance(image_paths, str):
        image_paths = [os.path.join(image_paths, img) for img in os.listdir(image_paths) if img.endswith('.png')]
    
    # sort image paths by name
    image_paths = sorted(image_paths)
    
    n, m = grid_size
    if image_size is None:
        with Image.open(image_paths[0]) as img:
            image_size = img.size

    width, height = image_size
    combined_image = Image.new('RGB', (m * width, n * height))

    for idx, img_path in enumerate(image_paths):
        if idx >= n * m:
            break
        with Image.open(img_path) as img:
            img = img.resize(image_size)
            row, col = divmod(idx, m)
            combined_image.paste(img, (col * width, row * height))

    combined_image.save(save_path)

def plot_metrics(
    metrics: dict, 
    save_path: str = None, 
    show: bool = False, 
    log_scale: bool = True, 
    bayes_optimal_accuracy=None,
    fig_title: str = "LeNetLifted Metrics"
):
    """
    Plot metrics in separate subplots for:
      Accuracy over Epochs (no shortcut/lifted),
      Accuracy - Shortcut Model,
      Accuracy - Lifted Model,
      Loss over Epochs (no CE/match),
      Loss - Prediction (CE),
      Loss - Matching
    
    Uses phase coloring for backgrounds, provides an optional figure title,
    and draws a Bayesian accuracy line if provided.
    """
    
    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], torch.Tensor):
                return [d.cpu().numpy() if isinstance(d, torch.Tensor) else d for d in data]
        return data
    
    # Convert all metrics to numpy arrays
    metrics = {k: to_numpy(v) for k, v in metrics.items()}

    # Style for each data split
    style_params = {
        'train': {'color': 'blue',  'linestyle': '-',  'alpha': 1.0},
        'val':   {'color': 'green', 'linestyle': ':',  'alpha': 1.0},
        'test':  {'color': 'purple','linestyle': '--', 'alpha': 1.0}
    }
    
    # Phase colors
    phase_colors = {
        'shortcut': '#FFB6C1',  # Light pink
        'lifted':   '#ADD8E6',  # Light blue
        'matching': '#98FB98'   # Light green
    }
    
    # Create figure with space for bottom legend
    fig = plt.figure(figsize=(20, 9))  # Slightly taller to accommodate legend
    gs = fig.add_gridspec(3, 3, height_ratios=[4, 4, 0.5])  # 2 rows for plots, 1 narrow row for legend
    
    # Main plots in 2x3 grid
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_acc_shortcut = fig.add_subplot(gs[0, 1])
    ax_acc_lifted = fig.add_subplot(gs[0, 2])
    ax_loss = fig.add_subplot(gs[1, 0])
    ax_loss_ce = fig.add_subplot(gs[1, 1])
    ax_loss_match = fig.add_subplot(gs[1, 2])
    
    # Phase legend subplot at bottom
    ax_phase_legend = fig.add_subplot(gs[2, :])
    ax_phase_legend.axis('off')
    
    def add_phase_background(ax, phases):
        """
        Add phase background coloring to a subplot without creating any labels.
        Returns None as no patches need to be tracked for legends.
        """
        current_phase = phases[0]
        start_idx = 0
        
        for i, phase in enumerate(phases):
            if phase != current_phase or i == len(phases) - 1:
                end_idx = i if phase != current_phase else i + 1
                ax.axvspan(start_idx, end_idx, 
                          alpha=0.2, 
                          color=phase_colors[current_phase])
                current_phase = phase
                start_idx = i
        return None

    # X-axis (epoch values) if available
    epochs = metrics.get("epoch", range(len(next(iter(metrics.values()))) if metrics else 0))
    
    # Separate out metrics by type
    for metric_name, values in metrics.items():
        # Skip meta keys
        if metric_name in ["epoch", "phases"]:
            continue
        if not values or all(v is None for v in values):
            continue
        
        # Determine train/val/test
        if 'train_' in metric_name:
            set_type = 'train'
            simple_key = metric_name.replace('train_', '')
        elif 'val_' in metric_name:
            set_type = 'val'
            simple_key = metric_name.replace('val_', '')
        elif 'test_' in metric_name:
            set_type = 'test'
            simple_key = metric_name.replace('test_', '')
        elif 'final_' in metric_name:
            # Skip final metrics for now
            continue
        else:
            # Fallback if none match
            set_type = 'train'
            simple_key = metric_name
        
        # ACCURACY
        if metric_name.endswith('_acc'):
            if metric_name.startswith('final_'):
                # Skip final accuracy for now
                continue
            # Decide which axis to use
            if 'shortcut_acc' in metric_name.lower():
                ax = ax_acc_shortcut
                ax_title = "Accuracy - Shortcut Model"
            elif 'lifted_acc' in metric_name.lower():
                ax = ax_acc_lifted
                ax_title = "Accuracy - Lifted Model"
            else:
                ax = ax_acc
                ax_title = "Accuracy over Epochs"
            
            ax.set_title(ax_title)
            ax.plot(
                epochs, 
                values, 
                label=f"{set_type.capitalize()} {simple_key}",
                **style_params[set_type]
            )
            if bayes_optimal_accuracy is not None and 'shortcut_acc' not in metric_name and 'lifted_acc' not in metric_name:
                # Only add Bayes line to main Accuracy subplot
                ax.axhline(y=bayes_optimal_accuracy, color='r', linestyle='--', label='Bayes Optimal', alpha=0.5)
        
        # LOSS
        elif metric_name.endswith('_loss'):
            if metric_name.startswith('final_'):
                # Skip final loss for now
                continue
            # Use log-scale if requested
            plot_vals = np.log(values) if log_scale else values
            if 'ce_loss' in metric_name.lower():
                ax = ax_loss_ce
                ax_title = "Loss - Prediction (ce)"
            elif 'match_loss' in metric_name.lower():
                ax = ax_loss_match
                ax_title = "Loss - Matching"
            else:
                ax = ax_loss
                ax_title = "Loss over Epochs"
            
            ax.set_title(ax_title)
            ax.plot(
                epochs, 
                plot_vals,
                label=f"{set_type.capitalize()} {simple_key}",
                **style_params[set_type]
            )
    
    # Add backgrounds for phases
    phase_patches = []  # Initialize phase_patches here
    phases = metrics.get('phases', None)
    if phases and len(phases) == len(epochs):
        # Add background coloring to all subplots (without labels)
        all_axes = [ax_acc, ax_acc_shortcut, ax_acc_lifted, ax_loss, ax_loss_ce, ax_loss_match]
        for ax in all_axes:
            add_phase_background(ax, phases)
        
        # Create separate patches just for the bottom legend
        for phase_name, color in phase_colors.items():
            patch = mpatches.Patch(color=color, alpha=0.2, 
                                 label=f"{phase_name.capitalize()} Phase")
            phase_patches.append(patch)
    
    if phase_patches:
        ax_phase_legend.legend(handles=phase_patches, 
                             title="Training Phases",
                             loc='center',
                             bbox_to_anchor=(0.5, 0.5),
                             ncol=len(phase_colors))  # Horizontal arrangement
    
    # Set axis labels, legends, grids
    for ax in [ax_acc, ax_acc_shortcut, ax_acc_lifted]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    for ax in [ax_loss, ax_loss_ce, ax_loss_match]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (log scale)" if log_scale else "Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space for legends
    
    # Save figure if requested
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    plt.close(fig)