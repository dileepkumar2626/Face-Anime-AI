from utils.checkpoint import (save_checkpoint,load_checkpoint,)
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from data.dataset import train_dataset
from models.cyclegan import CycleGAN
from utils.checkpoint import (save_checkpoint,load_checkpoint)
def train(
    num_epochs=1,
    batch_size=1,
    lr=2e-4,
    lambda_cycle=10.0,
    lambda_identity=0.5,
    checkpoint_dir="checkpoints",
    checkpoint_every=10,
    log_every=50,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = CycleGAN(
        lr=lr,
        lambda_cycle=lambda_cycle,
        lambda_identity=lambda_identity,
        device=device,
    )
    for epoch in range(1, num_epochs + 1):
        running = {}
        n_batches = 0
        for i, (real_face, real_anime) in enumerate(train_loader):
            log = model.train_step(real_face, real_anime)
            for k, v in log.items():
                running[k] = running.get(k, 0.0) + v
            n_batches += 1
            if (i + 1) % log_every == 0:
                avg = {k: v / n_batches for k, v in running.items()}
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch [{i + 1}/{len(train_loader)}] "
                    f"total: {avg['total']:.4f} | "
                    f"D_face: {avg['D_face']:.4f} | D_anime: {avg['D_anime']:.4f} | "
                    f"cycle: {avg['cycle']:.4f} | identity: {avg['identity']:.4f}"
                )
                running = {}
                n_batches = 0
        if epoch % checkpoint_every == 0 or epoch == num_epochs:
            save_checkpoint(model, epoch, checkpoint_dir)
    return model
if __name__ == "__main__":
    train(num_epochs=1)
 