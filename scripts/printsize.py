from src.datasets.ptbxl import PTBXLDataset
from src.datasets.ptbxl_ecg_demo import PTBXLECGDemoDataset


base_dir = "C:\\Users\\Administrator\\Desktop\\ptb-xl\\1.0.3"
classes = ["MI", "STTC", "HYP", "CD", "NORM"]

print("=== Baseline datasets ===")
train_base = PTBXLDataset(base_dir=base_dir, split="train", classes=classes)
val_base   = PTBXLDataset(base_dir=base_dir, split="val", classes=classes)
test_base  = PTBXLDataset(base_dir=base_dir, split="test", classes=classes)

print("Baseline train size =", len(train_base))
print("Baseline val size   =", len(val_base))
print("Baseline test size  =", len(test_base))

print("\n=== ECG+Demo datasets ===")
train_demo = PTBXLECGDemoDataset(base_dir=base_dir, split="train", classes=classes)
val_demo   = PTBXLECGDemoDataset(base_dir=base_dir, split="val", classes=classes)
test_demo  = PTBXLECGDemoDataset(base_dir=base_dir, split="test", classes=classes)

print("ECG+Demo train size =", len(train_demo))
print("ECG+Demo val size   =", len(val_demo))
print("ECG+Demo test size  =", len(test_demo))
