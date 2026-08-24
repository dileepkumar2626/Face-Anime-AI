import sys
from pathlib import Path
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.models.cyclegan import CycleGAN
from src.data.dataloader import train_data_loader
from src.utils.checkpoint import save_checkpoint
def train(num_epochs=30,lr=2e-4,lambda_cycle=10.0,lambda_identity=0.5,checkpoint_dir="checkpoints",log_every=50,device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    train_loader = train_data_loader
    model = CycleGAN(lr=lr,lambda_cycle=lambda_cycle,lambda_identity=lambda_identity,device=device,)
    best_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        epoch_total_loss = 0.0
        epoch_batches = 0
        running = {}
        n_batches = 0
        for i, (real_face, real_anime) in enumerate(train_loader):
            log = model.train_step(real_face,real_anime)
            epoch_total_loss += log["total"]
            epoch_batches += 1
            for k, v in log.items():
                running[k] = running.get(k, 0.0) + v
            n_batches += 1
            if (i + 1) % log_every == 0:
                avg = {k: v / n_batches for k, v in running.items()}
                print(
                    f"Epoch [{epoch}/{num_epochs}] "
                    f"Batch [{i + 1}/{len(train_loader)}] "
                    f"total: {avg['total']:.4f} | "
                    f"D_face: {avg['D_face']:.4f} | "
                    f"D_anime: {avg['D_anime']:.4f} | "
                    f"cycle: {avg['cycle']:.4f} | "
                    f"identity: {avg['identity']:.4f}"
                )
                running = {}
                n_batches = 0
        epoch_loss = (epoch_total_loss / epoch_batches)
        print(f"\nEpoch {epoch}/{num_epochs} finished")
        print(f"Average epoch loss: {epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(model,epoch,checkpoint_dir)
            print(f"*** NEW BEST MODEL SAVED ***")
            print(f"Epoch: {epoch}")
            print(f"Best loss: {best_loss:.4f}")
        else:
            print(f"Not better than best model.")
            print(f"Best loss: {best_loss:.4f}")
    return model
if __name__ == "__main__":
    train(num_epochs=30)