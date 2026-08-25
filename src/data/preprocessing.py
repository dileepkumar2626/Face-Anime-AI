import sys
from pathlib import Path
from PIL import Image,UnidentifiedImageError
from collections import Counter,defaultdict
from sklearn.model_selection import train_test_split
import hashlib
import json
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
anime_data_path = PROJECT_ROOT / "data" / "raw_data" / "anime"
real_data_path = PROJECT_ROOT / "data" / "raw_data" / "real"
print("Anime Dataset Exists:", anime_data_path.exists())
print("Contents (first 10):", list(anime_data_path.iterdir())[:10])
print("Real dataset Exists:", real_data_path.exists())
print("Contents (first 10):", list(real_data_path.iterdir())[:10])

def get_image_path(path):
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    images = [f for f in path.iterdir() if f.suffix.lower() in image_extensions]
    print(f"Found {len(images)} images")
    print(f"Total images: {len(images)}")
    return images
def get_corrupt_images(images):
    sizes = []              
    valid_images = []     
    corrupt_files = []
    images=images[:1000]
    for img_path in images:
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                sizes.append(img.size)   
            valid_images.append(img_path) 
        except (UnidentifiedImageError, OSError, Exception) as e:
            corrupt_files.append((img_path, str(e)))

    print(f"Valid images: {len(valid_images)}")
    print(f"Corrupt/unreadable images: {len(corrupt_files)}")

    if corrupt_files:
        print("\nCorrupt files found:")
        for path, err in corrupt_files[:20]:
            print(f"  {path.name} → {err}")

    size_counts = Counter(sizes)
    print("\nUnique sizes found:")
    for size, count in size_counts.most_common(10):
        print(f"{size}: {count} images")
    return valid_images,corrupt_files,size_counts
def dataset_split(valid_images):
    train,test=train_test_split(valid_images,test_size=0.2,random_state=42)
    return train,test
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
def remove_duplicate_images(image_paths):
    hash_groups = defaultdict(list)
    for img_path in image_paths:
        h = file_hash(img_path)
        hash_groups[h].append(img_path)
    deduped_paths = []
    removed_count = 0
    for h, paths in hash_groups.items():
        deduped_paths.append(paths[0])
        removed_count += len(paths) - 1
    print(f"Original count: {len(image_paths)}")
    print(f"Duplicates removed: {removed_count}")
    print(f"Final unique count: {len(deduped_paths)}")
    return deduped_paths
def save_dataset(path,filename):
    clean_paths_str = [str(p) for p in path]
    with open(filename, "w") as f:
        json.dump(clean_paths_str, f, indent=2)
    print(f"Saved {len(clean_paths_str)} paths to clean_image_paths.json")
anime_images=get_image_path(anime_data_path)
valid_anime_images,corrupt_anime_files,size_counts=get_corrupt_images(anime_images)
print(type(valid_anime_images))
deduplicated_anime_images=remove_duplicate_images(valid_anime_images)
anime_train,anime_test=dataset_split(deduplicated_anime_images)
save_dataset(anime_train,PROJECT_ROOT / "data" / "processed"/"anime_train"/"anime_train.json")
save_dataset(anime_test,PROJECT_ROOT / "data" / "processed"/"anime_test"/"anime_test.json")
real_images=get_image_path(real_data_path)
valid_real_images,corrupt_real_files,size_counts=get_corrupt_images(real_images)
print(type(valid_real_images))
deduplicated_real_images=remove_duplicate_images(valid_real_images)
real_train,real_test=dataset_split(deduplicated_real_images)
save_dataset(real_train,PROJECT_ROOT / "data" / "processed"/"real_train"/"real_train.json")
save_dataset(real_test,PROJECT_ROOT / "data" / "processed"/"real_test"/"real_test.json")
