import re
import hashlib
import numpy as np

def clean_concentration(val):
    """
    Parses numeric concentration values from metadata strings.
    Handles 'blank' as 0.0 and extracts leading floats.
    """
    val_str = str(val).strip().lower()
    if 'blank' in val_str:
        return 0.0
    match = re.search(r'[\d.]+', val_str)
    if match:
        return float(match.group())
    return np.nan

def get_file_hash(filepath):
    """
    Calculates the MD5 checksum of an image file to prevent 
    redundant retraining on identical strips.
    """
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()
