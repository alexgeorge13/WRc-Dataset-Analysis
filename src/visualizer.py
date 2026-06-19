import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def collect_comprehensive_stats(model_names, output_dir="data/outputs"):
    """Gathers all localized performance CSV logs from the project outputs directory."""
    all_dfs = []
    for name in model_names:
        csv_path = os.path.join(output_dir, f"{name}_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['Model'] = name
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else None


def generate_mscc5_comprehensive_table(df):
    """Cleans, aligns, and merges audit metrics and Top-K metrics across L3 and L4 splits."""
    if df is None:
        return None
    
    temp_df = df.copy()
    temp_df['Stage'] = temp_df['Hierarchy'].fillna(temp_df['Level'])

    for col in ['Model', 'Stage', 'Metric']:
        if col in temp_df.columns:
            temp_df[col] = temp_df[col].astype(str).str.strip()

    # Uniformly map both 'Code' and 'L4' variants to ensure 100% pivot intersection
    stage_map = {'Code': 'L4', 'code': 'L4', 'L4': 'L4', 'L3': 'L3', 'nan': np.nan}
    temp_df['Stage'] = temp_df['Stage'].map(stage_map)

    metric_cols = ['F1 (Macro)', 'F1 (Micro)', 'Value']
    for col in metric_cols:
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

    def get_stage_metrics(stage_name):
        subset = temp_df[temp_df['Stage'] == stage_name]
        f1_macro = subset.groupby('Model')['F1 (Macro)'].max()
        f1_micro = subset.groupby('Model')['F1 (Micro)'].max()
        tk_pivot = subset.pivot_table(index='Model', columns='Metric', values='Value', aggfunc='max')
        return pd.concat([f1_macro, f1_micro, tk_pivot], axis=1).reset_index()

    l3_metrics = get_stage_metrics('L3')
    l4_metrics = get_stage_metrics('L4')

    final = pd.DataFrame({'Model': temp_df['Model'].unique()})
    
    # Safely merge to prevent column drop errors if certain metrics are missing
    final = final.merge(l3_metrics[['Model', 'F1 (Macro)', 'F1 (Micro)', 'Top-3 Acc', 'Top-5 Acc']],
                        on='Model', how='left', suffixes=('', '_L3'))
    final = final.merge(l4_metrics[['Model', 'F1 (Macro)', 'F1 (Micro)', 'Top-3 Acc', 'Top-5 Acc']],
                        on='Model', how='left', suffixes=('_L3', '_L4'))

    final.columns = [
        'Model',
        'L3 F1-Macro', 'L3 F1-Micro (Acc)', 'L3 Top-3', 'L3 Top-5',
        'L4 F1-Macro', 'L4 F1-Micro (Acc)', 'L4 Top-3', 'L4 Top-5'
    ]
    return final.sort_values(by='L3 F1-Micro (Acc)', ascending=False)


def _render_clean_canvas(plot_df, y_col, hue_col, style_col, stage_col, title, ylabel, filename, save_dir):
    """Generates an individual plot with ultra-large fonts and clean, slimmed line paths."""
    level_map = {'L1': 1, 'L2': 2, 'L3': 3, 'L4': 4, 'Code': 4}
    
    df_canvas = plot_df.copy()
    df_canvas['x'] = df_canvas[stage_col].map(level_map)
    df_canvas = df_canvas.dropna(subset=['x', y_col])

    # Canvas dimensions preserved exactly at 8.5 x 6
    plt.figure(figsize=(8.5, 6))
    
    # Kept global font sizing at maximum scale 2.0
    sns.set_context("paper", font_scale=2.0)
    sns.set_style("whitegrid")

    unique_models = sorted(df_canvas['Model'].unique())
    palette = sns.color_palette("tab10", n_colors=len(unique_models))

    # --- Decreased line thickness and markers for a crisper chart layout ---
    plot_kwargs = {'marker': 'o', 'markersize': 8, 'linewidth': 2.0, 'palette': palette}
    if style_col:
        plot_kwargs['style'] = style_col

    sns.lineplot(data=df_canvas, x='x', y=y_col, hue=hue_col, **plot_kwargs)

    # Expanded all label assets to ultra-high visibility specifications
    plt.xticks([1, 2, 3, 4], ['L1\n(System)', 'L2\n(Group)', 'L3\n(Type)', 'L4\n(Code)'], fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.title(title, fontsize=18, fontweight='bold', pad=16)
    plt.ylabel(ylabel, fontsize=18, fontweight='bold')
    plt.xlabel("MSCC5 Hierarchy Level", fontsize=18, fontweight='bold')
    plt.ylim(-0.02, 1.05)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    
    # Drastically enlarged the external legend layout
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0, 
               frameon=True, fontsize=15, title_fontsize=16)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🚀 Saved Balanced Ultra-HD Image -> {save_path}")
    plt.close()


def generate_hierarchical_plots(df, save_dir="data/plots_ultra_hd"):
    """Generates the 5 standalone figures with maximized typography and clean line markers."""
    os.makedirs(save_dir, exist_ok=True)
    
    topk_rows = df[df['Metric'].notna() & df['Level'].notna()].copy()
    audit_rows = df[df['Hierarchy'].notna()].copy()

    # 1. Accuracy (Top-1 Only)
    top1_df = topk_rows[topk_rows['Metric'] == 'Top-1 Acc']
    _render_clean_canvas(
        top1_df, 'Value', 'Model', None, 'Level',
        "MSCC5 Accuracy (Top-1)", 
        "Accuracy Score", "fig1_accuracy_top1.png", save_dir
    )

    # 2. Accuracy (Top-1 and Top-3)
    top1_3_df = topk_rows[topk_rows['Metric'].isin(['Top-1 Acc', 'Top-3 Acc'])]
    _render_clean_canvas(
        top1_3_df, 'Value', 'Model', 'Metric', 'Level',
        "MSCC5 Accuracy (Top-1 vs Top-3)", 
        "Accuracy Score", "fig2_accuracy_top1_top3.png", save_dir
    )

    # 3. Accuracy (Top-1, Top-3, and Top-5)
    top1_3_5_df = topk_rows[topk_rows['Metric'].isin(['Top-1 Acc', 'Top-3 Acc', 'Top-5 Acc'])]
    _render_clean_canvas(
        top1_3_5_df, 'Value', 'Model', 'Metric', 'Level',
        "MSCC5 Support Envelope", 
        "Accuracy Score", "fig3_accuracy_top1_top3_top5.png", save_dir
    )

    # 4. Standalone Macro F1-Score
    _render_clean_canvas(
        audit_rows, 'F1 (Macro)', 'Model', None, 'Hierarchy',
        "MSCC5 Macro F1 Performance", 
        "Macro F1 Score", "fig4_f1_macro.png", save_dir
    )

    # 5. Standalone Micro F1-Score
    _render_clean_canvas(
        audit_rows, 'F1 (Micro)', 'Model', None, 'Hierarchy',
        "MSCC5 Micro F1 Performance", 
        "Micro F1 Score", "fig5_f1_micro.png", save_dir
    )