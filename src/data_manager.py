import os
import copy  # CRITICAL: Added to decouple shared dataset transformation pipelines
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, WeightedRandomSampler, DataLoader

def split_and_prepare_loaders(train_ds, folder_codes, metadata_dir="data/metadata", batch_size=32, train_transform=None, val_transform=None):
    """
    Handles robust stratified splitting for datasets with ultra-rare classes,
    saves/loads split indices for perfect persistence, breaks shared dataset references 
    to isolate training augmentations from evaluation steps, and builds PyTorch DataLoaders.
    """
    os.makedirs(metadata_dir, exist_ok=True)
    index_save_path = os.path.join(metadata_dir, "MSCC5_split_indices.pth")
    
    # 1. Get indices and labels for the entire dataset
    all_indices = list(range(len(train_ds)))
    all_labels = [sample[1] for sample in train_ds.samples]

    # Check if a persistent split configuration already exists locally
    if os.path.exists(index_save_path):
        print("📂 Found existing split indices locally. Loading to ensure 100% consistency...")
        indices = torch.load(index_save_path)
        train_indices = indices['train']
        val_indices = indices['val']
        test_indices = indices['test']
    else:
        print("🆕 No existing split found. Building robust stratified split splits...")
        
        # Identify ultra-rare classes with only one sample
        label_counts_full = np.bincount(all_labels, minlength=len(folder_codes))
        single_sample_classes_full = np.where(label_counts_full < 2)[0]

        stratifiable_indices = []
        stratifiable_labels = []
        non_stratifiable_indices = []

        for idx, label in zip(all_indices, all_labels):
            if label in single_sample_classes_full:
                non_stratifiable_indices.append(idx)
            else:
                stratifiable_indices.append(idx)
                stratifiable_labels.append(label)

        # First Split: 70% Train and 30% Temporary (Val + Test)
        train_strat_indices, temp_strat_indices = train_test_split(
            stratifiable_indices, test_size=0.30, 
            stratify=stratifiable_labels, random_state=42
        )
        train_indices = train_strat_indices + non_stratifiable_indices

        # Second Split: Split 30% Temporary into Val (1/3) and Test (2/3)
        temp_labels = [all_labels[i] for i in temp_strat_indices]
        label_counts_temp = np.bincount(temp_labels, minlength=len(folder_codes))
        single_sample_classes_temp = np.where(label_counts_temp < 2)[0]

        stratifiable_temp_indices = []
        stratifiable_temp_labels = []
        non_stratifiable_temp_indices = []

        for idx, label in zip(temp_strat_indices, temp_labels):
            if label in single_sample_classes_temp:
                non_stratifiable_temp_indices.append(idx)
            else:
                stratifiable_temp_indices.append(idx)
                stratifiable_temp_labels.append(label)

        val_strat_indices, test_strat_indices = train_test_split(
            stratifiable_temp_indices, test_size=0.6667, 
            stratify=stratifiable_temp_labels, random_state=42
        )
        
        train_indices += non_stratifiable_temp_indices
        val_indices = val_strat_indices
        test_indices = test_strat_indices

        # Save indices for reproducibility across setups
        torch.save({'train': train_indices, 'val': val_indices, 'test': test_indices}, index_save_path)
        print(f"✅ Saved split indices to: {index_save_path}")

    # 2. Subsets Creation
    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(train_ds, val_indices)
    test_subset = Subset(train_ds, test_indices)

    # =====================================================================
    # TRANSFORMATION DECOUPLING BLOCK
    # =====================================================================
    if train_transform is not None and val_transform is not None:
        # Shallow copy the root dataset wrapper so validation and testing 
        # do not accidentally execute training augmentations at runtime
        val_subset.dataset = copy.copy(train_ds)
        test_subset.dataset = copy.copy(train_ds)
        
        # Inject the isolated processing configurations
        train_subset.dataset.preprocess = train_transform
        val_subset.dataset.preprocess = val_transform
        test_subset.dataset.preprocess = val_transform

    # 3. Create the Weighted Random Sampler for handling class imbalance
    train_labels_only = [all_labels[i] for i in train_indices]
    class_counts = np.bincount(train_labels_only, minlength=len(folder_codes))
    
    # Avoid division by zero on missing classes
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    weights[torch.isinf(weights)] = 0
    sample_weights = [weights[l] for l in train_labels_only]
    train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # 4. Initialize DataLoaders (num_workers=0 is stable across Windows/VS Code)
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_indices, val_indices, test_indices


def generate_dataset_manifest(train_ds, folder_codes, train_indices, val_indices, test_indices, df_mapping, metadata_dir="data/metadata"):
    """Generates an audit-ready dataset manifest tracking split memberships and hierarchies."""
    manifest_path = os.path.join(metadata_dir, "MSCC5_Full_Dataset_Manifest.csv")
    
    if os.path.exists(manifest_path):
        print(f"✅ Manifest already exists at: {manifest_path}. Skipping compilation.")
        return pd.read_csv(manifest_path)
        
    print("🆕 Compiling dataset manifest mapping file...")
    split_map = {idx: 'Train' for idx in train_indices}
    for idx in val_indices: split_map[idx] = 'Validation'
    for idx in test_indices: split_map[idx] = 'Test'

    manifest_data = []
    for i in range(len(train_ds)):
        path, label_idx = train_ds.samples[i]
        manifest_data.append({
            'Dataset_Index': i,
            'Filename': os.path.basename(path),
            'Folder_Code': folder_codes[label_idx],
            'Split': split_map.get(i, 'Unassigned'),
            'Full_Path': os.path.abspath(path)
        })

    manifest_df = pd.DataFrame(manifest_data)
    
    # Clean up and merge structural hierarchies
    mapping_clean = df_mapping.copy()
    mapping_clean['L1'] = mapping_clean['L1'].astype(str).str.split('.').str[0].str.strip()
    
    manifest_df = manifest_df.merge(
        mapping_clean[['Code', 'L1', 'L2', 'L3']],
        left_on='Folder_Code', right_on='Code', how='left'
    ).drop(columns=['Code'])

    manifest_df.to_csv(manifest_path, index=False)
    print(f"✅ Full Dataset Manifest exported to: {manifest_path}")
    return manifest_df