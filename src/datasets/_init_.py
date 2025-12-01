"""
Docstring for datasets._init_
Expose the PTBXLDataset class as part of the datasets package.

THis file allows external modules to import the dataset with:
    From src.datasets import PTBXLDataset
The implementation of the dataset (loading, validation, normalization, etc.)
is kept in ptbxl.py to maintain a clean and modular project structure.

"""

from .ptbxl import PTBXLDataset
