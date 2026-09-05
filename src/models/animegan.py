import torch
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

class AnimeGANv2:
    def __init__(
        self,
        device=None,
        checkpoint_path=PROJECT_ROOT/'checkpoints'/"anime2_v2.pth",
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = torch.hub.load(
            "bryandlee/animegan2-pytorch:main",
            "generator",
            pretrained="face_paint_512_v2",
            device=self.device,
        ).eval()
        checkpoint_path = Path(checkpoint_path)
        torch.save(
            self.model.state_dict(),
            checkpoint_path
        )

        print(
            f"AnimeGANv2 model saved to: {checkpoint_path}"
        )
        self.face2paint = torch.hub.load(
            "bryandlee/animegan2-pytorch:main",
            "face2paint",
            size=512,
            device=self.device,
        )
    @torch.no_grad()
    def predict(self, image):
        """image: PIL.Image (RGB) -> returns PIL.Image"""
        return self.face2paint(self.model, image)
AnimeGANv2()