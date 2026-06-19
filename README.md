# Sewer Defect Classification Framework

A modular deep learning benchmarking framework for classifying sewer defects using the standard Water Research Centre (WRc) Dataset matching the Manual of Sewer Condition Classification 5th Edition (MSCC5) guidelines. 

This repository implements, trains, and evaluates standard deep vision architectures (e.g., CNNs and Vision Transformers) across hierarchical defect taxonomies.

## 📂 Repository File Structure

```text
SewerDefectClassification/
│
├── .cache/model_weights/            # Offline backbone storage (.pth weights for timm)
├── data/
│   ├── metadata/                    # Master asset sheets, taxonomies, and split states
│   │   ├── MSCC5_Defect_Code_Groups.csv
│   │   ├── MSCC5_Full_Dataset_Manifest.csv
│   │   ├── MSCC5_split_indices.pth
│   ├── outputs/                     # Compiled CSV ledgers & manuscript tables
│   │   ├── FINAL_BENCHMARK_RESULTS.csv
│   │   ├── ...
│   │   └── MANUSCRIPT_LEADERBOARD_TABLE.csv
│   ├── plots/                       # Model classification performance curves
│   │   ├── fig1_accuracy_top1.png
│   │   ├── ...
│   │   └── fig5_f1_micro.png
│   └── WRcDataset/                  # 72 asset categories matching MSCC5 (e.g., B, CC, DEF)
│
├── src/                             # Core Modular Production Package
│   ├── models/factory.py            # Wrap/resize weights to match custom project classification heads
│   ├── dataset.py                   # PyTorch dataset parsing & bicubic transformations
│   ├── data_manager.py              # Stratified partitioning & Outlier class shielding
│   ├── engine.py                    # Training hooks & verification metrics loops
│   ├── utils.py                     # Deterministic random seeds & environment setups
│   └── visualizer.py                # Graph engines for plotting validation metrics
│
├── models/checkpoints/              # Fine-tuned, completely unfrozen sewer model weights
│   ├── resnet50_standard_Full_Unfrozen.pth
│   └── swin_standard_Full_Unfrozen.pth
│
├── tests/test_gpu.py                # Baseline CUDA availability test
├── main.py                          # Primary benchmark, ledger compilation, and audit engine
├── evaluate.py                      # Post-Run Analysis, Generate Plots
└── requirements.txt                 # Environment dependency file
```
## 📥 Dataset Acquisition Guide

The benchmark image library represents a cross-sector collaboration of 7 UK water companies, compiling over 27,000 manually validated images mapping across 72 strict MSCC5 defect codes.

### How to access the raw data:
Because the dataset is hosted securely within the UK Water Sector's closed platforms, follow this specific routing to acquire the files:

1. **Read Project Info:** Read the project context at the WRc Group Case Study Portal ([https://www.wrcgroup.com/resources/case-studies/ofwat-innovation-challenge-ai-and-sewer-defects-analysis/](https://www.wrcgroup.com/resources/case-studies/ofwat-innovation-challenge-ai-and-sewer-defects-analysis/)).
2. **Account Creation:** Visit Spring Innovation ([https://spring-innovation.co.uk/](https://spring-innovation.co.uk/)), click to create a free user account, and log in.
3. **The Mirror Jump:** Once logged in, you will be redirected to the underlying portal at: [https://ukwir2021.my.site.com/spring/s/](https://ukwir2021.my.site.com/spring/s/)
4. **The Secondary Login Trick:** On the UKWIR MySite page, you must click "Log In" a second time to register your session state fully inside the data portal.
5. **Search & Download:** In the main portal search bar, look for the official project repository name: "United Utilities: Artificial Intelligence and Sewers"
6. **Local Deployment:** Download the zip payload, extract it, and place the raw code subfolders inside your local `data/WRcDataset/` directory.
7. **Data Clubbing:** Consolidate all the folders that are split numerically (e.g., merging `RM_1` and `RM_2` into a single `RM` parent directory).

---

## 🛠️ SETUP AND INSTALLATION WORKSPACE OPERATIONS

Follow these exact operational steps inside your terminal to stand up an identical local environment, initialize dependencies, and verify your local GPU card.

### STEP 1: Create a Raw Virtual Environment
To bypass pre-release interpreter hooks, build a clean virtual environment without the bundled package installer:
```
python -m venv venv --without-pip
```


### STEP 2: Activate the Environment
On Windows PowerShell:
```
.\venv\Scripts\Activate
```

On Mac/Linux:
```
source venv/bin/activate
```


### STEP 3: Bootstrap and Update Pip Manually
Now that the isolated space is active, download and inject a stable standalone pip build:
```
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
del get-pip.py
```


### STEP 4: Install Core Framework Requirements
```
pip install -r requirements.txt
```


### STEP 5: Override with Dedicated GPU PyTorch Binaries 
By default, standard pip pulls a CPU-only PyTorch setup on Windows. Run these commands to explicitly swap it for the heavy CUDA-enabled system binaries:
```
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```


### STEP 6: Verify Local Hardware Access
Execute the diagnostic test script to confirm that PyTorch can successfully communicate with your local graphics card:
```
python tests/test_gpu.py
```


===================================================================


## 🚀 EXECUTION DRIVERS RUN ROUTINES

### 1. Main Benchmarking and Training Ledger
```
python main.py
```
Parses the 72 structural subfolders, verifies model layer dimensions, skips already completed backbones using a bypass shield, trains remaining networks over a 10-epoch fine-tuning cycle, and compiles model performances.


### 2. Model Evaluation 
```
python evaluate.py
```
Primary verification engine that processes model checkpoints against the validation and test splits to compute multi-level hierarchical metrics, classification ledgers, and top-K accuracy diagnostics.




---
