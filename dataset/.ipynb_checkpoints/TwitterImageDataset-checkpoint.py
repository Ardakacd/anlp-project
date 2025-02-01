import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random


class TwitterImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # Load control group
        control_path = os.path.join(root_dir, "control_group/users")
        print(control_path)
        self.load_images(control_path, label=0)

        # Load diagnosed group
        diagnosed_path = os.path.join(root_dir, "diagnosed_group/users")
        print(diagnosed_path)
        self.load_images(diagnosed_path, label=1)

        # Shuffle data for randomness
        random.shuffle(self.data)

    def load_images(self, group_path, label):
        for user in os.listdir(group_path):  
            user_folder = os.path.join(group_path, user, "images")
            if os.path.exists(user_folder):  
                for img_file in os.listdir(user_folder):
                    img_path = os.path.join(user_folder, img_file)
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):  
                        self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')  
        if self.transform:
            image = self.transform(image)  
        return image, torch.tensor(label, dtype=torch.long)