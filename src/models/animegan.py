import torch
class AnimeGANv2:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
 
        self.model = torch.hub.load(
            "bryandlee/animegan2-pytorch:main",
            "generator",
            pretrained="face_paint_512_v2",
            device=self.device,
        ).eval()
 
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