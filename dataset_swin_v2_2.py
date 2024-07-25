import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


class BirdDataset(Dataset):
    def __init__(
        self,
        dataframe,
        transform=None,
        use_upscaled=True,
        is_test=False,
        label_to_idx=None,
    ):
        self.dataframe = dataframe
        self.transform = transform
        self.use_upscaled = use_upscaled
        self.is_test = is_test
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if self.use_upscaled:
            img_path = self.dataframe.iloc[idx]["upscale_img_path"]
        else:
            img_path = self.dataframe.iloc[idx]["img_path"]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image
        else:
            label = self.dataframe.iloc[idx]["label"]
            label_idx = self.label_to_idx[label]  # self.label_to_idx 사용
            return image, label_idx


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)


def prepare_datasets(csv_path, test_size=0.1, random_state=42):
    df = pd.read_csv(csv_path)
    train_df, valid_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    labels = df["label"].unique()
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    return train_df, valid_df, label_to_idx, idx_to_label
