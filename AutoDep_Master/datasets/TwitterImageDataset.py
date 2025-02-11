import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class TwitterImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Get the absolute path of the directory where this script resides
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the absolute path to the dataset root directory
        self.root_dir = os.path.abspath(os.path.join(script_dir, root_dir))
        self.transform = transform
        self.data = []

        # Load control group images
        control_path = os.path.join(self.root_dir, "control_group", "users")
        print(f"Loading control group from: {control_path}")
        self.load_images(control_path, label=0)

        # Load diagnosed group images
        diagnosed_path = os.path.join(self.root_dir, "diagnosed_group", "users")
        print(f"Loading diagnosed group from: {diagnosed_path}")
        self.load_images(diagnosed_path, label=1)

        # Shuffle data for randomness
        random.shuffle(self.data)

        print(f"Dataset successfully loaded with {len(self.data)} images.")

    def load_images(self, group_path, label):
        if not os.path.exists(group_path):
            print(f"Warning: {group_path} does not exist.")
            return

        for user in os.listdir(group_path):
            # Skip hidden files and directories
            if user.startswith('.'):
                continue

            user_folder = os.path.join(group_path, user, "images")
            if os.path.exists(user_folder):
                for img_file in os.listdir(user_folder):
                    # Skip hidden files
                    if img_file.startswith('.'):
                        continue

                    img_path = os.path.join(user_folder, img_file)
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.data.append((img_path, label))
            else:
                print(f"Warning: {user_folder} does not exist.")

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
    # Pass the relative path from the script's directory to the dataset root
    dataset = TwitterImageDataset("../data")
    print("Dataset initialized successfully.")
    print(f"Total images in dataset: {len(dataset)}")
