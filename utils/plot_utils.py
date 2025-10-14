import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_losses(train_losses, test_losses, save_dir, exp_tag):
    """
    Plot training and test losses over epochs and save to file.
    
    Parameters:
    -----------
    train_losses : list or np.ndarray
        Training losses for each epoch
    test_losses : list or np.ndarray
        Test losses for each epoch
    save_dir : str or Path
        Directory to save the plot
    exp_tag : str
        Experiment tag to include in the filename
    
    Returns:
    --------
    str
        Path to the saved plot
    """
    # Ensure save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Plot losses
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if test_losses is not None and len(test_losses) > 0:
        ax.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    
    # Formatting
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training and Test Losses - {exp_tag}', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    save_path = save_dir / f"{exp_tag}_losses.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return str(save_path)
