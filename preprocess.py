import torch
from torchvision import transforms
from typing import Tuple
import pandas as pd
from PIL import Image, ImageOps
import os

PATH_COLS = ["image_path", "filepath", "image", "path", "file_name"]
LAT_COLS = ["Latitude", "latitude", "lat"]
LON_COLS = ["Longitude", "longitude", "lon"]

# preprocess data similar to how data is preprocessed in ResNet-18
# reference: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
tf = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),  # scaling to [0.0,1.0]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def find_col_name(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No column {candidates} found in {list(df.columns)}")

def load_image(path: str):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)  # rotate if necessary
        return tf(img.convert("RGB"))

def prepare_data(csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Template preprocessing for leaderboard.

    Requirements:
    - Must read the provided data path at `path`.
    - Must return a tuple (X, y):
        X: a list of model-ready inputs (these must match what your model expects in predict(...))
        y: a list of ground-truth labels aligned with X (same length)

    Notes:
    - The evaluation backend will call this function with the shared validation data
    - Ensure the output format (types, shapes) of X matches your model's predict(...) inputs.
    """
    df = pd.read_csv(csv_path)  # read csv
    base_dir = os.path.dirname(os.path.abspath(csv_path))   # get directory where csv and images live

    path_col = find_col_name(df, PATH_COLS)
    lat_col = find_col_name(df, LAT_COLS)
    lon_col = find_col_name(df, LON_COLS)

    X = torch.stack([load_image(os.path.join(base_dir, p)) for p in df[path_col]])   # get transformed image
    y = torch.tensor(df[[lat_col, lon_col]].values, dtype=torch.float32)
    return X, y