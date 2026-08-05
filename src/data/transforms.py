from torchvision import transforms
from PIL import Image
def transform_train_data(image_size):
    train_transform = transforms.Compose([
    transforms.Resize(int(image_size * 1.12), Image.BICUBIC),
    transforms.RandomCrop(image_size),                        
    transforms.RandomHorizontalFlip(),                        
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return train_transform
def transform_test_data(image_size):
    test_transform = transforms.Compose([
    transforms.Resize(int(image_size * 1.12), Image.BICUBIC),                                                 # faces are ~symmetric
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return test_transform