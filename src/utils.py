import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Sets random seeds across libraries for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed locked at: {seed}")

def save_checkpoint(model, name, save_dir="models/checkpoints"):
    """Saves model weights locally."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.pth")
    torch.save(model.state_dict(), path)
    print(f"✅ Saved model weights to {path}")