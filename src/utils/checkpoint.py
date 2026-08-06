from pathlib import Path
import torch
def save_checkpoint(model, epoch, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"cyclegan_epoch_{epoch}.pth"
    torch.save(
        {
            "epoch": epoch,
            "G_face2anime": model.G_face2anime.state_dict(),
            "G_anime2face": model.G_anime2face.state_dict(),
            "D_face": model.D_face.state_dict(),
            "D_anime": model.D_anime.state_dict(),
            "optimizer_G": model.optimizer_G.state_dict(),
            "optimizer_D_face": model.optimizer_D_face.state_dict(),
            "optimizer_D_anime": model.optimizer_D_anime.state_dict(),
        },
        path,
    )
    print(f"Checkpoint saved -> {path}")
def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=model.device,
    )
    model.G_face2anime.load_state_dict(
        checkpoint["G_face2anime"]
    )
    model.G_anime2face.load_state_dict(
        checkpoint["G_anime2face"]
    )
    model.D_face.load_state_dict(
        checkpoint["D_face"]
    )
    model.D_anime.load_state_dict(
        checkpoint["D_anime"]
    )
    model.optimizer_G.load_state_dict(
        checkpoint["optimizer_G"]
    )
    model.optimizer_D_face.load_state_dict(
        checkpoint["optimizer_D_face"]
    )
    model.optimizer_D_anime.load_state_dict(
        checkpoint["optimizer_D_anime"]
    )
    print("Checkpoint Loaded")
    return checkpoint["epoch"]