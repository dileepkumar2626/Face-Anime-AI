from PIL import Image
from torch.utils.data import Dataset
import json
import random
from pathlib import Path
from transforms import transform_train_data,transform_test_data
class CycleGANDataset(Dataset):
    def __init__(self, face_json_paths, anime_json_paths, transform=None):
        self.transform = transform
        self.real_paths = []
        for jp in face_json_paths:
            with open(jp, "r") as f:
                self.real_paths.extend(json.load(f))
        if not self.real_paths:
                    raise ValueError("Face dataset is empty")
        self.anime_paths = []
        for jp in anime_json_paths:
            with open(jp, "r") as f:
                self.anime_paths.extend(json.load(f))
        if not self.anime_paths:
                    raise ValueError("Anime dataset is empty")
    def __len__(self):
        return max(len(self.real_paths), len(self.anime_paths))
    def __getitem__(self, idx):
        real_path = self.real_paths[idx % len(self.real_paths)]
        anime_path = self.anime_paths[random.randint(0, len(self.anime_paths) - 1)]
        real_img = Image.open(real_path).convert("RGB")
        anime_img = Image.open(anime_path).convert("RGB")
        if self.transform:
            real_img = self.transform(real_img)
            anime_img = self.transform(anime_img)
        return real_img, anime_img
PROJECT_ROOT = Path(__file__).resolve().parents[2]
train_transform=transform_train_data(256)
train_dataset=CycleGANDataset(face_json_paths=[
        PROJECT_ROOT / "data" / "processed" / "real_train" / "real_train.json"],
        anime_json_paths=[PROJECT_ROOT / "data" / "processed" / "anime_train" / "anime_train.json"],transform=train_transform)
test_transform=transform_test_data(256)
test_dataset=CycleGANDataset(face_json_paths=[
        PROJECT_ROOT / "data" / "processed" / "real_test" / "real_test.json"],
        anime_json_paths=[PROJECT_ROOT / "data" / "processed" / "anime_test" / "anime_test.json"],transform=test_transform)