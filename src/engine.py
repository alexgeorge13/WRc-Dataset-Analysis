import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from src.utils import save_checkpoint

def perform_comprehensive_audit(model, test_loader, m_df, device='cuda'):
    """
    Collects predictions and maps targets back to structural hierarchies
    using explicit MSCC5 code layout definitions.
    """
    model.eval()
    y_true_indices = []
    y_pred_indices = []

    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            _, preds = torch.max(out, 1)

            y_true_indices.extend(lbls.cpu().tolist())
            y_pred_indices.extend(preds.cpu().tolist())

    # 1. Vectorized row slicing from our known columns matrix
    true_df = m_df.iloc[y_true_indices].reset_index(drop=True).add_suffix('_True')
    pred_df = m_df.iloc[y_pred_indices].reset_index(drop=True).add_suffix('_Pred')
    full_audit_df = pd.concat([true_df, pred_df], axis=1)

    # 2. Compute explicit validation metrics for each tier of the MSCC5 hierarchy
    summary_stats = []
    for level in ['L1', 'L2', 'L3', 'L4']:
        # Extract the Series cleanly using our verified layout strings
        y_t = full_audit_df[f'{level}_True'].to_numpy()
        y_p = full_audit_df[f'{level}_Pred'].to_numpy()

        acc = accuracy_score(y_t, y_p)
        prec, rec, f1, _ = precision_recall_fscore_support(y_t, y_p, average='macro', zero_division=0)
        micro_f1 = f1_score(y_t, y_p, average='micro', zero_division=0)

        summary_stats.append({
            'Hierarchy': level,
            'Accuracy': acc,
            'Precision (Macro)': prec,
            'Recall (Macro)': rec,
            'F1 (Macro)': f1,
            'F1 (Micro)': micro_f1
        })

    return full_audit_df, pd.DataFrame(summary_stats)


def verify_and_get_topk_stats(model, model_name, test_loader, m_df, k_list=[1, 3, 5], device='cuda'):
    """Projects L4/Code probabilities onto parent hierarchical groups to calculate Top-K."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(lbls.cpu())

    probs_all = torch.cat(all_probs)
    labels_all = torch.cat(all_labels)

    results = []
    hierarchies = ['L1', 'L2', 'L3', 'Code']

    for level in hierarchies:
        unique_vals = m_df[level].unique()
        num_classes_at_level = len(unique_vals)
        val_to_idx = {val: i for i, val in enumerate(unique_vals)}

        # Structural Probability Projection Matrix [72 x Classes_at_Level]
        projection_matrix = torch.zeros(72, num_classes_at_level)
        for l4_idx in range(72):
            level_val = m_df.iloc[l4_idx][level]
            level_idx = val_to_idx[level_val]
            projection_matrix[l4_idx, level_idx] = 1.0

        level_probs = torch.matmul(probs_all, projection_matrix)
        level_labels_true = torch.tensor([val_to_idx[m_df.iloc[l.item()][level]] for l in labels_all])

        for k in k_list:
            if k >= num_classes_at_level:
                acc = 1.0
            else:
                _, topk_preds = torch.topk(level_probs, k, dim=1)
                correct = torch.any(topk_preds == level_labels_true.unsqueeze(1), dim=1).sum().item()
                acc = correct / len(labels_all)

            results.append({
                'Model': model_name,
                'Level': 'L4' if level == 'Code' else level,
                'Metric': f'Top-{k} Acc',
                'Value': acc
            })

    return pd.DataFrame(results)


def train_model_full(model, model_name, train_loader, val_loader, epochs=10, device='cuda'):
    """Unfreezes backbones and executes full optimization with Label Smoothing and OneCycleLR."""
    print(f"\n🔥 Starting FULL TRAINING: {model_name} ({epochs} Epochs, Unfrozen)")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-4,
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    save_checkpoint(model, f"{model_name}_Full_Unfrozen")
    return model