import os
import shutil
import random

# Paths
train_dir = "dataset/train"
val_dir = "dataset/validation"

# Split ratio (keep 20% for validation)
split_ratio = 0.8

# Make sure validation is empty first
if os.path.exists(val_dir):
    shutil.rmtree(val_dir)
os.makedirs(val_dir)

for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)
    split_point = int(len(images) * split_ratio)

    # Split images
    val_images = images[split_point:]

    # Create class folder in validation
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Move 20% images to validation
    for img in val_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_dir, class_name, img)
        shutil.move(src, dst)

print("✅ Dataset split done! 80% in train / 20% in validation.")
