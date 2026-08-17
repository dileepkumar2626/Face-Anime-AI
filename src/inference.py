import argparse
import torch
from torchvision import transforms
from PIL import Image
from models.generator import ResnetGenerator

IMAGE_SIZE = 256
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
denormalize = transforms.Normalize(
    mean=[-1.0, -1.0, -1.0],
    std=[2.0, 2.0, 2.0],
)
def load_generator(checkpoint_path, device):
    """Load only G_face2anime from a full CycleGAN checkpoint."""
    generator = ResnetGenerator(input_nc=3, output_nc=3, n_residual_blocks=9).to(device)
 
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["G_face2anime"] if "G_face2anime" in checkpoint else checkpoint
    generator.load_state_dict(state_dict)
 
    generator.eval()  
    return generator
def run_inference(generator, image_path, device):
    """Load one image, apply test transforms, generate the anime version."""
    image = Image.open(image_path).convert("RGB")
    input_tensor = test_transform(image).unsqueeze(0).to(device)  
    with torch.no_grad():
        fake_anime = generator(input_tensor)
    return fake_anime.squeeze(0).cpu() 
def save_image(tensor, output_path):
    """Denormalize a [-1, 1] tensor back to [0, 1] and save as an image file."""
    tensor = denormalize(tensor).clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(output_path)
    print(f"Saved: {output_path}")
def main():
    parser = argparse.ArgumentParser(description="CycleGAN face -> anime inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--input", type=str, required=True, help="Path to input face image")
    parser.add_argument("--output", type=str, default="anime_result.jpg", help="Where to save the output image")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = load_generator(args.checkpoint, device)
    fake_anime = run_inference(generator, args.input, device)
    save_image(fake_anime, args.output)
    
if __name__ == "__main__":
    main()