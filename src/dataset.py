import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SewerDataset(Dataset):
    def __init__(self, root_dir, folder_codes, preprocess):
        """
        Args:
            root_dir (str): Path to the folder containing image subfolders (e.g., 'data/WRcDataset')
            folder_codes (list): List of MSCC5 code strings (e.g., ['CL', 'CC', ...])
            preprocess (callable): CLIP/Vision transform pipeline
        """
        self.root_dir = root_dir
        self.folder_codes = folder_codes
        self.preprocess = preprocess
        self.samples = []

        # Sort folder codes and image names for strict consistency across runs
        for idx, code in enumerate(sorted(folder_codes)):
            folder_path = os.path.join(root_dir, code)
            if os.path.isdir(folder_path):
                for img_name in sorted(os.listdir(folder_path)):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(folder_path, img_name), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx = self.samples[idx]
        try:
            image = self.preprocess(Image.open(img_path))
            return image, label_idx
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            # Fallback block to avoid breaking the batch
            return torch.zeros((3, 224, 224)), label_idx