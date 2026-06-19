import os
import pandas as pd
from src.visualizer import (
    collect_comprehensive_stats, 
    generate_mscc5_comprehensive_table, 
    generate_hierarchical_plots
)

def run_analysis_pipeline():
    print("📋 Initializing Post-Run Research Analysis Engine...")
    
    # Models to test for comprehensive evaluation and leaderboard generation
    model_names_to_test = [
        "resnet50_standard",
        "swin_standard", 
        "vit_small",
        "vit_tiny",
        "mobilenetv4_edge"
    ]
    
    output_dir = "data/outputs"
    plots_dir = "data/plots"
    
    # 1. Collect all local evaluation logs
    master_df = collect_comprehensive_stats(model_names_to_test, output_dir=output_dir)
    
    if master_df is None:
        print(f"⚠️ Error: No baseline result files found in '{output_dir}'. Please execute 'main.py' first.")
        return

    print("✅ Comprehensive stats database compiled smoothly.")

    # 2. Render and output the 3-panel high-resolution paper graphics
    print("📈 Generating multi-panel hierarchical evaluation trends...")
    generate_hierarchical_plots(master_df, save_dir=plots_dir)

    # 3. Formulate your paper leaderboard metrics matrix
    print("\n🏆 MSCC5 ARCHITECTURE COMPARISON: PERFORMANCE LEADERBOARD")
    winners_df = generate_mscc5_comprehensive_table(master_df)
    
    # Render stylized terminal summary block
    print("-" * 110)
    print(winners_df.to_string(
        index=False,
        formatters={
            'L3 F1-Macro': '{:.3f}'.format, 'L3 F1-Micro (Acc)': '{:.3f}'.format,
            'L3 Top-3': '{:.1%}'.format, 'L3 Top-5': '{:.1%}'.format,
            'L4 F1-Macro': '{:.3f}'.format, 'L4 F1-Micro (Acc)': '{:.3f}'.format,
            'L4 Top-3': '{:.1%}'.format, 'L4 Top-5': '{:.1%}'.format
        }
    ))
    print("-" * 110)
    
    # 4. Export clean results file for verification
    summary_csv_path = os.path.join(output_dir, "MANUSCRIPT_LEADERBOARD_TABLE.csv")
    winners_df.to_csv(summary_csv_path, index=False)
    print(f"\n💾 Production-ready table saved to: {summary_csv_path}")


if __name__ == "__main__":
    # This keeps it clean, unambiguous, and completely distinct from main.py
    run_analysis_pipeline()