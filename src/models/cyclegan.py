import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.models.generator import ResnetGenerator
from src.models.discriminator import PatchDiscriminator
from src.models.losses import CycleGANLosses
class CycleGAN(nn.Module):
    def __init__(
        self,
        lr=2e-4,
        beta1=0.5,
        beta2=0.999,
        lambda_cycle=10.0,
        lambda_identity=0.5,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.G_face2anime = ResnetGenerator(
            input_nc=3,
            output_nc=3,
            n_residual_blocks=9,
        ).to(device)
        self.G_anime2face = ResnetGenerator(
            input_nc=3,
            output_nc=3,
            n_residual_blocks=9,
        ).to(device)
        self.D_face = PatchDiscriminator().to(device)
        self.D_anime = PatchDiscriminator().to(device)
        self.losses = CycleGANLosses(
            lambda_cycle=lambda_cycle,
            lambda_identity=lambda_identity,
            device=device,
        )
        self.optimizer_G = optim.Adam(
            list(self.G_face2anime.parameters()) +
            list(self.G_anime2face.parameters()),
            lr=lr,
            betas=(beta1, beta2),
        )
        self.optimizer_D_face = optim.Adam(
            self.D_face.parameters(),
            lr=lr,
            betas=(beta1, beta2),
        )
        self.optimizer_D_anime = optim.Adam(
            self.D_anime.parameters(),
            lr=lr,
            betas=(beta1, beta2),
        )
    def train_step(self, real_face, real_anime):
        real_face = real_face.to(self.device)
        real_anime = real_anime.to(self.device)
        fake_anime = self.G_face2anime(real_face)
        fake_face = self.G_anime2face(real_anime)
        reconstructed_face = self.G_anime2face(fake_anime)
        reconstructed_anime = self.G_face2anime(fake_face)
        identity_face = self.G_anime2face(real_face)
        identity_anime = self.G_face2anime(real_anime)
        self.optimizer_G.zero_grad()
        g_loss, log = self.losses.compute_generator_loss(
            self.D_face,
            self.D_anime,
            real_face,
            real_anime,
            fake_anime,
            fake_face,
            reconstructed_face,
            reconstructed_anime,
            identity_face,
            identity_anime,
        )
        g_loss.backward()
        self.optimizer_G.step()
        self.optimizer_D_face.zero_grad()
        d_face_loss = self.losses.compute_discriminator_loss(
            self.D_face,
            real_face,
            fake_face,
        )
        d_face_loss.backward()
        self.optimizer_D_face.step()
        self.optimizer_D_anime.zero_grad()
        d_anime_loss = self.losses.compute_discriminator_loss(
            self.D_anime,
            real_anime,
            fake_anime,
        )
        d_anime_loss.backward()
        self.optimizer_D_anime.step()
        return {
            **log,
            "D_face": d_face_loss.item(),
            "D_anime": d_anime_loss.item(),
        }
if __name__ == "__main__":
    from src.data.dataset import train_dataset
    from src.data.dataloader import train_data_loader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = train_data_loader
    model = CycleGAN(device=device)
    real_face, real_anime = next(iter(train_loader))
    print("=" * 50)
    print("Input Shapes")
    print("=" * 50)
    print("Real Face :", real_face.shape)
    print("Real Anime:", real_anime.shape)

    losses = model.train_step(real_face, real_anime)

    print("\n" + "=" * 50)
    print("Losses")
    print("=" * 50)

    for k, v in losses.items():
        print(f"{k}: {v:.4f}")