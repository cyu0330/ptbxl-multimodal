from src.datasets.ptbxl import PTBXLDataset
from src.datasets.ptbxl_ecg_multimodal import PTBXLECGMultimodalDataset

# Base directory and class list
base_dir = "C:\\Users\\Administrator\\Desktop\\ptb-xl\\1.0.3"
classes = ["MI", "STTC", "HYP", "CD", "NORM"]

print("=== Baseline datasets ===")
train_base = PTBXLDataset(base_dir=base_dir, split="train", classes=classes)
val_base   = PTBXLDataset(base_dir=base_dir, split="val",   classes=classes)
test_base  = PTBXLDataset(base_dir=base_dir, split="test",  classes=classes)

print("Baseline train size:", len(train_base))
print("Baseline val size:  ", len(val_base))
print("Baseline test size: ", len(test_base))

print("\n=== ECG + Demographics datasets ===")
train_mm = PTBXLECGMultimodalDataset(base_dir=base_dir, split="train", classes=classes)
val_mm   = PTBXLECGMultimodalDataset(base_dir=base_dir, split="val",   classes=classes)
test_mm  = PTBXLECGMultimodalDataset(base_dir=base_dir, split="test",  classes=classes)

print("ECG+Demo train size:", len(train_mm))
print("ECG+Demo val size:  ", len(val_mm))
print("ECG+Demo test size: ", len(test_mm))
