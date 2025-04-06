import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class TwitterImageDataset(Dataset):
    def __init__(self, root_dir="AutoDep_Master/data", transform=None):
        # Dynamically resolve the absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script location
        project_root = os.path.abspath(os.path.join(script_dir, "../../"))  # Move up to anlp-project

        # Normalize path to fix Windows/macOS inconsistencies
        self.root_dir = os.path.normpath(os.path.join(project_root, root_dir))
        self.transform = transform
        self.data = []

        # Debugging: Print only main dataset path
        print(f"Resolved dataset path: {self.root_dir}")

        # Load control group images
        control_path = os.path.join(self.root_dir, "control_group", "users")
        print("Accessing control group data...")
        self.load_images(control_path, label=0)

        # Load diagnosed group images
        diagnosed_path = os.path.join(self.root_dir, "diagnosed_group", "users")
        print("Accessing diagnosed group data...")
        self.load_images(diagnosed_path, label=1)

        random.shuffle(self.data)
        print(f"✅ Dataset successfully loaded with {len(self.data)} images.")

    def load_images(self, group_path, label):
        group_path = os.path.normpath(group_path)

        if not os.path.exists(group_path):
            print(f"⚠️ Warning: {group_path} does not exist.")
            return

        for user in os.listdir(group_path):
            if user.startswith('.') or user in ["Thumbs.db", "desktop.ini"]:
                continue

            user_folder = os.path.normpath(os.path.join(group_path, user))
            images_folder = os.path.join(user_folder, "images")

            if os.path.exists(images_folder) and os.path.isdir(images_folder):
                for img_file in os.listdir(images_folder):
                    if img_file.startswith('.'):
                        continue

                    img_path = os.path.normpath(os.path.join(images_folder, img_file))
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    print("Initializing TwitterImageDataset...")
    dataset = TwitterImageDataset()
    print("Dataset initialized successfully.")
    print(f"Total images in dataset: {len(dataset)}")
