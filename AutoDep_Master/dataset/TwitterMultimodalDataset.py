import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import re

class TwitterMultimodalDataset(Dataset):
    def __init__(self, root_dir="AutoDep_Master/data", transform=None, max_tokens=512):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "../../"))
        self.root_dir = os.path.normpath(os.path.join(project_root, root_dir))
        self.transform = transform
        self.max_tokens = max_tokens
        self.data = []

        print(f"📁 Resolved dataset path: {self.root_dir}")
        
        control_path = os.path.join(self.root_dir, "control_group", "users")
        diagnosed_path = os.path.join(self.root_dir, "diagnosed_group", "users")

        print("🔍 Accessing control group data...")
        self.load_users(control_path, label=0)

        print("🔍 Accessing diagnosed group data...")
        self.load_users(diagnosed_path, label=1)

        print(f"✅ Multimodal dataset successfully loaded with {len(self.data)} entries.")

    def find_image_with_extensions(self, folder, base_filename, extensions=(".jpg", ".jpeg", ".png")):
        for ext in extensions:
            candidate = os.path.join(folder, base_filename + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def load_users(self, group_path, label):
        if not os.path.exists(group_path):
            print(f"⚠️ Group path does not exist: {group_path}")
            return

        for user in os.listdir(group_path):
            if user.startswith('.') or user in ["Thumbs.db", "desktop.ini"]:
                continue

            user_folder = os.path.join(group_path, user)
            images_folder = os.path.join(user_folder, "images")

            profile_path = self.find_image_with_extensions(images_folder, f"{user}_profile")
            banner_path = self.find_image_with_extensions(images_folder, f"{user}_banner")
            tweet_path = os.path.join(user_folder, f"{user}_Tweets.csv")

            has_images = profile_path is not None and banner_path is not None
            has_text = os.path.exists(tweet_path)

            if has_images and has_text:
                self.data.append({
                    "profile_img": profile_path,
                    "banner_img": banner_path,
                    "tweet_path": tweet_path,
                    "label": label
                })
            else:
                print(f"⚠️ Missing image or text data for user: {user}")

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r"[^a-z.!? ]+", "", text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        profile_img = Image.open(entry["profile_img"]).convert("RGB")
        banner_img = Image.open(entry["banner_img"]).convert("RGB")

        if self.transform:
            profile_img = self.transform(profile_img)
            banner_img = self.transform(banner_img)

        try:
            df = pd.read_csv(entry["tweet_path"])
            texts = df["full_text"].dropna().tolist()
        except Exception as e:
            print(f"❌ Error reading {entry['tweet_path']}: {e}")
            texts = []

        texts = [self.preprocess_text(t) for t in texts if not t.startswith("RT ")]
        joined_text = " ".join(texts)[:self.max_tokens]

        return {
            "profile_img": profile_img,
            "banner_img": banner_img,
            "text": joined_text,
            "label": torch.tensor(entry["label"], dtype=torch.long)
        }

if __name__ == "__main__":
    print("🔄 Initializing TwitterMultimodalDataset...")
    dataset = TwitterMultimodalDataset()
    print("✅ Dataset initialized successfully.")
    print(f"📦 Total multimodal entries in dataset: {len(dataset)}")

