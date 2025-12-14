# PTB-XL Multimodal ECG Framework 
This repository contains the code developed for an MSc dissertation project Based on the PTB-XL ECG datasets.

The project includes:
- a baseline CNN model for multi-label ECG classification (MI, STTC, HYP, CD, NORM);
- a multimodal extension combining ECG signals with demographic features;
- a binary atrial fibrillation (AF) classification task;
- Grad-CAM-based visualisation for model interpretation; and
- a lightweight demo for quick inference and visualisation.


The codebase is organised to support reproducible training, evaluation, and interpretation experiments, and to allow examiners to either run lightweight demos or reproduce full training results using the official PTB-XL data.

## 1. Environment Setup

The codebase is written in Python and has been tested with Python 3.9.
A virtual environment is recommended to avoid dependency conflicts.

Create a virtual environment in the project root:

```bash
python -m venv .venv
```  
Activate the environment.
On Windows:

```bash
.\.venv\Scripts\activate
```  
On macOS or Linux:

```bash
source .venv/bin/activate
```  
Install the required packages:

```bash
pip install -r requirements.txt
```  
## 2. Quick Demo (No PTB-XL Required)
The demo uses a pretrained model obtained from prior training on the
PTB-XL dataset, following the standard patient-wise splits.
A small ECG sample is provided for demonstration purposes so that the
full inference and explainability pipeline can be run without additional
data preparation.

ECG sample stored in the `data/demo/` directory.
Run the following command from the project root:

```bash
python scripts/00_demo_inference.py --demo_path data/demo/single/single_sample_00.npz --class_idx 0 --lead 0
```
The script will print the predicted probabilities in the terminal.
Grad-CAM figures will be saved to the following directory:
```text
outputs/demo/
```
The --class_idx argument can be changed to visualise different
diagnostic classes.

## Full Pipeline
The quick demo shown above uses a pretrained model that was obtained
during earlier experiments. The steps below are only needed if you
would like to run the full pipeline starting from the raw data.

### Dataset Download
This project uses the PTB-XL dataset, which is a publicly available ECG
dataset provided via PhysioNet.

The dataset can be found at:

https://physionet.org/content/ptb-xl/1.0.3/

Te data can be downloaded using:

```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```
The dataset path should then be set in the `base_dir` field of the
configuration files.

### Training
#### Baseline ECG Model:
A baseline 1D CNN is implemented for multi-label ECG classification
using the PTB-XL diagnostic superclasses.
Training can be started with:

```bash
python scripts/03_train_ecg_baseline.py --config configs/ecg_baseline.yaml
```
#### Multimodal Model
A simple multimodal extension is included, where demographic features
are combined with ECG representations.
The model can be trained using:
```bash
python scripts/04_train_multimodal_prototype.py --config configs/ecg_multimodal.yaml
```
#### AF Binary Classification
An additional binary classification task is implemented for atrial
fibrillation detection.
```bash
python scripts/05_train_af_binary.py --config configs/af_binary.yaml
```
### Testing
After training, each model is evaluated on the test split using separate
testing scripts. These scripts load the saved checkpoints and generate
prediction results without retraining the models.

Baseline ECG model testing:
```bash
python scripts/06_ecg_baseline_test.py --config configs/ecg_baseline.yaml --ckpt outputs/ecg_baseline/ckpts/ecg_baseline_best.pth --out_csv outputs/ecg_baseline/preds/ecg_baseline_test_preds.csv --threshold 0.5
```
Multimodal ECG + demographics model testing:
```bash
```bash
python scripts/07_ecg_multimodal_test.py --config configs/ecg_multimodal.yaml --ckpt outputs/ecg_multimodal/ckpts/ecg_multimodal_best.pth --out_csv outputs/ecg_multimodal/preds/ecg_multimodal_test_preds.csv
```
AF binary classification testing:
```bash
python scripts/08_af_binary_test.py --config configs/af_binary.yaml --ckpt outputs/af_binary/ckpts/af_binary_best.pth --out_csv outputs/af_binary/preds/af_binary_test_preds.csv
```
### Merged Results and Analysis
To support comparison across different models, prediction results from
multiple test runs can be merged into a single table.
The following script merges all test predictions:
```bash
python scripts/09_merge_all_test.py
```
Basic analysis on the merged results can then be performed using:
```bash
python scripts/10_analyse_merged_test.py
```
Merged prediction tables are saved under:

```text
outputs/merged/
```

### Explainability (Grad-CAM)
Grad-CAM is used to provide visual explanations for the CNN-based ECG
models by highlighting time regions that contribute most to a given
prediction.

Grad-CAM visualisations are generated using separate scripts for
different tasks.


Baseline ECG model:
```bash
python scripts/11_grad_cam_ecg_baseline.py
```
Multimodal ECG model:
```bash
python scripts/12_grad_cam_ecg_demo.py
```
AF binary classification:
```bash
python scripts/13_grad_cam_af.py
```
Grad-CAM figures are saved under:
```text
outputs/gradcam/
outputs/gradcam_demo/
outputs/gradcam_af/
```
### Plotting and Visualisation
Several scripts are provided to generate figures for result analysis and
reporting.

These include:
- overall performance summaries,
- prediction distributions, and
- model-specific comparison plots.

Main plotting scripts:
```bash
python scripts/14_plot_results.py
python scripts/15_plot_distributions.py
```
Additional plots for individual models:
```bash
python scripts/16_plot_baseline_only.py
python scripts/17_plot_mm_only.py
```

### Outputs
All outputs generated during training, testing, and analysis are saved
under the `outputs/`

```text
outputs/
├── af_binary/        # outputs for AF binary classification
├── ecg_baseline/     # outputs for baseline ECG model
├── ecg_demo/         # outputs for demo / lightweight models
├── ecg_multimodal/   # outputs for multimodal ECG + demographics model
│
├── ckpts/            # saved model checkpoints
├── figures/          # plots and summary figures used in the report
├── merged/           # merged prediction tables for analysis
│
├── gradcam/          # Grad-CAM outputs for baseline and multimodal models
├── gradcam_af/       # Grad-CAM outputs for AF classification
├── gradcam_demo/     # Grad-CAM outputs generated by the demo script
│
└── demo/             # quick demo outputs (single-sample inference)

