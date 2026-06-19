import sys
import os
import torch
import pandas as pd
import torchvision.transforms as T

sys.path.append('.')

from src.utils import set_seed
from src.dataset import SewerDataset
from src.data_manager import split_and_prepare_loaders, generate_dataset_manifest
from src.models.factory import get_model
from src.engine import train_model_full, perform_comprehensive_audit, verify_and_get_topk_stats

# =====================================================================
# CUSTOM TRANSFORM: PRESERVE ASPECT RATIO VIA SQUARE PADDING
# =====================================================================
class SquarePad(object):
    """
    Pads the shorter side of a rectangular image symmetrically with black pixels 
    to create a perfect 1:1 square canvas, preserving the native aspect ratio 
    of the sewer pipe visual features without warping, squishing, or cropping out edges.
    """
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        # Padding configuration: (left, top, right, bottom)
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return T.functional.pad(image, padding, 0, 'constant')


def run_model_diagnostics(model_names, folder_codes, device):
    print("\n🔍 Launching Benchmarking Architecture Diagnostics...")
    num_classes = len(folder_codes)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    report = []

    for name in model_names:
        print(f"Testing Architecture: {name}...", end=" ")
        try:
            model = get_model(name, num_classes, device=device)
            model.eval()

            with torch.no_grad():
                output = model(dummy_input)

            expected_shape = (2, num_classes)
            if output.shape == expected_shape:
                print("✅ PASSED")
                report.append((name, "Success", output.shape))
            else:
                print(f"❌ SHAPE MISMATCH (Got {output.shape})")
                report.append((name, "Error", f"Mismatched Dimensions: {output.shape}"))

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"🔥 BLOCKING FAILURE: {str(e)}")
            report.append((name, "Error", str(e)))

    print("\n--- Model Benchmark Architecture Ledger ---")
    for name, status, detail in report:
        print(f"{name:<25} | {status:<10} | {detail}")

    return all(r[1] == "Success" for r in report)


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    csv_path = "data/metadata/MSCC5_Defect_Code_Groups.csv"
    dataset_dir = "data/WRcDataset"
    output_dir = "data/outputs"
    checkpoint_dir = "models/checkpoints"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if not os.path.exists(csv_path) or not os.path.exists(dataset_dir):
        print(f"❌ Error: Please ensure data files are populated inside your local 'data/' directory.")
        return

    print("\n--- Initializing Dataset Hierarchical Taxonomy ---")
    hierarchy_df = pd.read_csv(csv_path)
    
    code_col = next((c for c in hierarchy_df.columns if 'code' in c.lower() or 'folder' in c.lower()), hierarchy_df.columns[0])
    folder_codes = list(hierarchy_df[code_col].astype(str).str.strip().unique())

    print("\n--- Establishing Dynamic Dual-Pipeline Transformations ---")
    
    # 🛡️ COMBINED DOMAIN-SAFE SEWER TRAINING PIPELINE (No Flips, Optimized Spatial Step)
    train_transform = T.Compose([
        SquarePad(),  # Retains aspect ratio perfectly (Always safe)
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        
        # 🔄🗺️ COMBINED STEP: Handles both minor roll (±10°) and controlled shifting (10%)
        # in a single mathematical pass, preventing double-interpolation blur.
        T.RandomApply([
            T.RandomAffine(
                degrees=(-10, 10),           # Subtle camera roll
                translate=(0.1, 0.1),     # Max 10% horizontal/vertical shift
                scale=None,                 # Keeps zoom at 100% (no warping)
                shear=None,                 # No skewing/distortion
                fill=0                      # Fills empty margins with matching black
            )
        ], p=0.5),
        
        # --- Photometric Augmentations (Safe for all orientation-sensitive classes) ---
        T.ColorJitter(
            brightness=0.2,   # Simulates lighting fluctuations from LED rings
            contrast=0.2,     # Mimics reflections off wet pipe surfaces vs dark concrete
            saturation=0.1    # Accounts for murky vs clear water tinting
        ),
        
        # Sharpness adjustments help models identify cracks under blurry lighting
        T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
        
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Pipeline 2: Evaluation Configuration (Pad -> Resize -> Clean Deterministic Formats Only)
    val_test_transform = T.Compose([
        SquarePad(),  # Matches training scale proportions exactly
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Root dataset initialized with training configurations as base reference
    dataset = SewerDataset(root_dir=dataset_dir, folder_codes=folder_codes, preprocess=train_transform)

    print("\n--- Establishing Train/Val/Test Partitions ---")
    # UPDATED: Passing explicit independent pipelines to your modified data manager
    train_loader, val_loader, test_loader, train_idx, val_idx, test_idx = split_and_prepare_loaders(
        dataset, 
        folder_codes, 
        batch_size=32,
        train_transform=train_transform,
        val_transform=val_test_transform
    )

    print("\n--- Generating Dataset Manifest Ledger ---")
    manifest_df = generate_dataset_manifest(dataset, folder_codes, train_idx, val_idx, test_idx, hierarchy_df)

    print("\n--- Split Membership Summary ---")
    print(manifest_df.groupby('Split').size().reset_index(name='Count'))

    model_names_to_test = [
        "resnet50_standard",
        "swin_standard",
        "vit_small",
        "vit_tiny",
        "mobilenetv4_edge"
    ]
    
    if not run_model_diagnostics(model_names_to_test, folder_codes, device):
        print("\n⚠️ Architecture diagnostics failure detected. Halting run.")
        return
        
    print("\n🚀 All backbones healthy. Moving directly to training loop execution...")
    
    final_full_results = []

    for name in model_names_to_test:
        output_csv = os.path.join(output_dir, f"{name}_results.csv")
        checkpoint_path = os.path.join(checkpoint_dir, f"{name}_Full_Unfrozen.pth")

        # 🔄 RESTORED: Full Results Bypass Shield
        if os.path.exists(output_csv):
            print(f"\n📊 Found complete evaluation ledger for {name} at: {output_csv}")
            print(f"➡️ Skipping execution loop entirely for this backbone.")
            combined_model_report = pd.read_csv(output_csv)
            final_full_results.append(combined_model_report)
            continue

        # Re-initialize basic model shell structure only if results don't exist
        model = get_model(name, len(folder_codes), device=device)

        # Checkpoint Persistence Shield for Training Loop
        if os.path.exists(checkpoint_path):
            print(f"\n✨ Found completed weights for {name} locally.")
            print(f"💾 Loading from checkpoint: {checkpoint_path} -> Skipping Training Loop.")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"\n⌐ Training Final Model: {name}")
            model = train_model_full(model, name, train_loader, val_loader, epochs=10, device=device)

        print(f"⌐ Running Detailed Audit: {name}")
        _, summary_df = perform_comprehensive_audit(model, test_loader, hierarchy_df, device=device)

        print(f"⌐ Calculating Top-K Stats: {name}")
        topk_df = verify_and_get_topk_stats(model, name, test_loader, hierarchy_df, device=device)

        summary_df['Model'] = name
        topk_df['Model'] = name
        combined_model_report = pd.concat([summary_df, topk_df], ignore_index=True)
        combined_model_report.to_csv(output_csv, index=False)
        final_full_results.append(combined_model_report)

        del model
        torch.cuda.empty_cache()

    if final_full_results:
        master_final_df = pd.concat(final_full_results, ignore_index=True)
        master_output_path = os.path.join(output_dir, "FINAL_BENCHMARK_RESULTS.csv")
        master_final_df.to_csv(master_output_path, index=False)

        print(f"\n✨ ALL MODELS COMPLETE. Summary results saved to: {master_output_path}")
    else:
        print("⚠️ No execution results compiled to process.")

if __name__ == "__main__":
    main()