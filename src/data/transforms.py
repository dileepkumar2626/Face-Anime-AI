from torchvision import transforms
from PIL import Image
def transform_data(image_size):
    transform = transforms.Compose([
    transforms.Resize(int(image_size * 1.12), Image.BICUBIC),
    transforms.RandomCrop(image_size),                        
    transforms.RandomHorizontalFlip(),                        
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform