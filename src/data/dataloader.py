from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.dataset import train_dataset,test_dataset
from torch.utils.data import DataLoader
train_data_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)
for real, anime in train_data_loader:
    print(real.shape)
    print(anime.shape)
    print(real.dtype)
    print(anime.dtype)
    break
test_data_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)
for real, anime in test_data_loader:
    print(real.shape)
    print(anime.shape)
    print(real.dtype)
    print(anime.dtype)
    break
