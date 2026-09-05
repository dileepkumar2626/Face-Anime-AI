from torchvision import transforms
from PIL import Image
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(CASCADE_PATH)
def crop_face(
    image: Image.Image,
    margin: float = 0.25,
) -> Image.Image:
    """
    Detect the largest face and crop it with an additional margin.

    Args:
        image: PIL RGB image.
        margin: Extra area around the detected face.
                0.25 means 25% additional margin.

    Returns:
        Cropped square PIL image.

    Raises:
        ValueError: If no face is detected.
    """
    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        raise ValueError("No face detected in the uploaded image.")
    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )
    margin_x = int(w * margin)
    margin_y = int(h * margin)

    left = max(0, x - margin_x)
    top = max(0, y - margin_y)
    right = min(image_rgb.width, x + w + margin_x)
    bottom = min(image_rgb.height, y + h + margin_y)

    cropped = image_rgb.crop(
        (left, top, right, bottom)
    )
    width, height = cropped.size
    size = max(width, height)
    square = Image.new(
        "RGB",
        (size, size),
        (255, 255, 255),
    )
    paste_x = (size - width) // 2
    paste_y = (size - height) // 2

    square.paste(
        cropped,
        (paste_x, paste_y),
    )

    return square
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
    transforms.Resize((image_size, image_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return test_transform