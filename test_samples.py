import os
import sys
import glob
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as T

sys.path.append('.')

# Pull modules directly from your project foundation
from src.models.factory import get_model

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
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return T.functional.pad(image, padding, 0, 'constant')


# 📋 PATH & MODEL CONFIGURATION
SAMPLES_DIR = "data/sample_test_images"  
TAXONOMY_CSV_PATH = "data/metadata/MSCC5_Defect_Code_Groups.csv"
OUTPUT_DIR = "data/outputs"
OUTPUT_PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "sample_set_top3_predictions.csv")
CHECKPOINT_DIR = "models/checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🤖 EVALUATION TARGET ARCHITECTURES
MODEL_NAMES_TO_TEST = [
    "resnet50_standard",
    "swin_standard",
    "vit_small",
    "vit_tiny",
    "mobilenetv4_edge"
]

# 🛡️ Unified inference transform pipeline
INFERENCE_TRANSFORMS = T.Compose([
    SquarePad(),
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_class_mapping_list():
    """Parses hierarchy file to map numeric nodes via sorted alphanumeric standard abbreviations."""
    if not os.path.exists(TAXONOMY_CSV_PATH):
        raise FileNotFoundError(f"❌ Ledger missing at: {TAXONOMY_CSV_PATH}")
    
    df = pd.read_csv(TAXONOMY_CSV_PATH)
    df.columns = df.columns.str.strip()
    
    # Use the 'Code' column (abbreviation) as the true structural sorting key
    code_key_col = 'Code' if 'Code' in df.columns else df.columns[0]
    
    # Alphabetize the codes to match the raw numeric class output indices assigned during training
    sorted_codes = sorted(list(df[code_key_col].astype(str).str.strip().unique()))
    return sorted_codes


def main():
    print("🎬 Starting Multi-Model Top-3 Hierarchical Inference Engine...")
    
    # 1. Parse Class Indexes from CSV
    try:
        sorted_abbrev_codes = load_class_mapping_list()
        num_classes = len(sorted_abbrev_codes)
    except Exception as e:
        print(f"🔥 Failed to load taxonomy mapping: {str(e)}")
        return
    
    # 2. Discover Image Targets
    supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    for ext in supported_extensions:
        image_paths.extend(glob.glob(os.path.join(SAMPLES_DIR, ext)))
        
    if not image_paths:
        print(f"⚠️ No test images found inside '{SAMPLES_DIR}'. Please populate sample files.")
        return
        
    sorted_image_paths = sorted(image_paths)
    print(f"📷 Found {len(sorted_image_paths)} sample images to analyze (Taxonomy Classes: {num_classes}).")
    
    # Load taxonomy dataframe for explicit direct column matching
    df_taxonomy = pd.read_csv(TAXONOMY_CSV_PATH)
    df_taxonomy.columns = df_taxonomy.columns.str.strip()
    
    # Target exact columns from your updated schema structure
    code_col = 'Code'
    l1_col, l2_col, l3_col, l4_col = 'L1', 'L2', 'L3', 'L4'

    # Master list to collect prediction dictionaries for the final CSV save
    all_predictions_records = []

    # 3. Outer Loop: Model Audit & Verification
    for model_name in MODEL_NAMES_TO_TEST:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_Full_Unfrozen.pth")
        
        print(f"\n⚙️ Evaluating Model Architecture: {model_name}")
        print(f" └─ Verifying trained weight status...")
        
        if not os.path.exists(checkpoint_path):
            print(f"    ❌ SKIPPED: Trained checkpoint file not found at: '{checkpoint_path}'")
            continue
            
        print(f"    💾 SUCCESS: Checkpoint found. Loading explicit {num_classes}-class network architecture...")
        try:
            model = get_model(model_name, num_classes=num_classes, device=DEVICE)
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            model.eval()
        except Exception as e:
            print(f"    🔥 Model initialization failure: {str(e)}")
            continue
            
        # 📊 STRUCTURED MONITOR DISPLAY
        print(f"\n    {'Image Filename':<16} | {'Rank':<4} | {'L1':<16} | {'L2':<20} | {'L3':<16} | {'L4 [Code]':<32} | {'Confidence':<10}")
        print("    " + "=" * 146)
        
        # 4. Inner Loop: Evaluate individual test images extracting top 3 paths
        for path in sorted_image_paths:
            filename = os.path.basename(path)
            try:
                raw_img = Image.open(path).convert('RGB')
                tensor_img = INFERENCE_TRANSFORMS(raw_img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits = model(tensor_img)
                    probabilities = torch.softmax(logits, dim=1)[0]
                    
                    # Extract top 3 highest probabilities and their matching index IDs
                    topk_confidences, topk_indices = torch.topk(probabilities, k=3)
                
                # Process each rank row dynamically
                for rank_idx in range(3):
                    conf = topk_confidences[rank_idx].item()
                    pred_idx = topk_indices[rank_idx].item()
                    
                    predicted_code_abbrev = sorted_abbrev_codes[pred_idx]
                    
                    # Direct lookups matching back to your 'Code' tracker row
                    matched_rows = df_taxonomy[df_taxonomy[code_col].astype(str).str.strip() == predicted_code_abbrev]
                    
                    if not matched_rows.empty:
                        row = matched_rows.iloc[0]
                        l1_val = str(row[l1_col]).strip() if l1_col in df_taxonomy.columns else "N/A"
                        l2_val = str(row[l2_col]).strip() if l2_col in df_taxonomy.columns else "N/A"
                        l3_val = str(row[l3_col]).strip() if l3_col in df_taxonomy.columns else "N/A"
                        l4_val = str(row[l4_col]).strip() if l4_col in df_taxonomy.columns else "N/A"
                    else:
                        l1_val, l2_val, l3_val, l4_val = "Unknown", "Unknown", "Unknown", "Unknown"
                    
                    composite_l4_string = f"{l4_val} ({predicted_code_abbrev})" if l4_val != "Unknown" else f"Unknown ({predicted_code_abbrev})"

                    # Record results to predictions storage array
                    all_predictions_records.append({
                        "Model": model_name,
                        "Image Filename": filename,
                        "Rank": rank_idx + 1,
                        "L1": l1_val,
                        "L2": l2_val,
                        "L3": l3_val,
                        "L4": composite_l4_string,
                        "Confidence": f"{conf:.4f}"
                    })

                    # Safeguard terminal alignments from long textual descriptions
                    l1_disp = (l1_val[:14] + '..') if len(l1_val) > 16 else l1_val
                    l2_disp = (l2_val[:18] + '..') if len(l2_val) > 20 else l2_val
                    l3_disp = (l3_val[:14] + '..') if len(l3_val) > 16 else l3_val
                    l4_disp = (composite_l4_string[:29] + '..') if len(composite_l4_string) > 32 else composite_l4_string
                    
                    # Only display filename on the first rank row to keep visual layout scannable
                    file_display_name = filename if rank_idx == 0 else ""
                        
                    print(f"    {file_display_name:<16} | #{rank_idx+1:<3} | {l1_disp:<16} | {l2_disp:<20} | {l3_disp:<16} | {l4_disp:<32} | {conf:>10.2%}")
                
                print("    " + "-" * 146)  # Subtle divider between images
                
            except Exception as e:
                print(f"    {filename:<16} | ❌ Processing Error: {str(e)}")
                print("    " + "-" * 146)
                
        print("    " + "=" * 146)
        del model
        torch.cuda.empty_cache()

    # =====================================================================
    # SAVE RAW RECORDS TO THE OUTPUT CSV FILE MATRIX
    # =====================================================================
    if all_predictions_records:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_df = pd.DataFrame(all_predictions_records)
        output_df.to_csv(OUTPUT_PREDICTIONS_CSV, index=False)
        print(f"\n💾 SUCCESS: All model top-3 predictions saved to: {OUTPUT_PREDICTIONS_CSV}")
    else:
        print("\n⚠️ No predictions collected. Output execution canceled.")

if __name__ == "__main__":
    main()