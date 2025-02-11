# Ensure necessary NLTK resources are downloaded
"""
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import re

class TwitterTextDataset(Dataset):
    def __init__(self, root_dir):
        # Get the absolute path of the directory where this script resides
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the absolute path to the dataset root directory
        self.root_dir = os.path.abspath(os.path.join(script_dir, root_dir))
        self.data = []

        # Load control group texts
        control_path = os.path.join(self.root_dir, "control_group", "users")
        print(f"Loading control group from: {control_path}")
        self.load_texts(control_path, label=0, user_limit = 5)

        # Load diagnosed group texts
        diagnosed_path = os.path.join(self.root_dir, "diagnosed_group", "users")
        print(f"Loading diagnosed group from: {diagnosed_path}")
        self.load_texts(diagnosed_path, label=1, user_limit = 5)

        # Shuffle data for randomness
        random.shuffle(self.data)

        print(f"Dataset successfully loaded with {len(self.data)} text entries.")

    def load_texts(self, group_path, label, user_limit=None):
        if not os.path.exists(group_path):
            print(f"Warning: {group_path} does not exist.")
            return

        user_count = 0  # Initialize a counter for processed users

        for user in os.listdir(group_path):
            # Skip hidden files and directories
            if user.startswith('.'):
                continue

            user_folder = os.path.join(group_path, user)
            if os.path.exists(user_folder):
                for file in os.listdir(user_folder):
                    # Skip hidden files
                    if file.startswith('.'):
                        continue

                    file_path = os.path.join(user_folder, file)
                    if file.lower().endswith('.csv'):
                        self.process_csv(file_path, label)

                user_count += 1  # Increment the user counter

                # Check if the user limit has been reached
                if user_limit is not None and user_count >= user_limit:
                    print(f"Processed {user_count} users; reached the specified limit.")
                    return
            else:
                print(f"Warning: {user_folder} does not exist.")


    def process_csv(self, file_path, label):
        try:
            df = pd.read_csv(file_path)
            # Extract text from all columns and rows
            for column in df.columns:
                for text in df[column].dropna():
                    processed_text = self.preprocess_text(str(text))
                    self.data.append((processed_text, label))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = text.strip()
        return text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        return text, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    print("Initializing TwitterTextDataset...")
    dataset = TwitterTextDataset("../data")
    print("Dataset initialized successfully.")
    print(f"Total texts in dataset: {len(dataset)}")

