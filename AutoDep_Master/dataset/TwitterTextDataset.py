import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")  # Load ALBERT tokenizer

class TwitterTextDataset(Dataset):
    def __init__(self, root_dir, max_tokens=512):
        # Get the absolute path of the directory where this script resides
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.normpath(os.path.join(script_dir, root_dir))
        self.data = []
        self.max_tokens = max_tokens  # Maximum number of tokens per text

        # Debugging: Show resolved dataset path
        print(f"Resolved dataset path: {self.root_dir}")

        # Load control group texts
        control_path = os.path.join(self.root_dir, "control_group", "users")
        print("Accessing control group data...")
        self.load_texts(control_path, label=0, user_limit=10)

        # Load diagnosed group texts
        diagnosed_path = os.path.join(self.root_dir, "diagnosed_group", "users")
        print("Accessing diagnosed group data...")
        self.load_texts(diagnosed_path, label=1, user_limit=10)

        # Shuffle data for randomness
        random.shuffle(self.data)

        print(f"✅ Dataset successfully loaded with {len(self.data)} text entries.")

    def load_texts(self, group_path, label, user_limit=None):
        if not os.path.exists(group_path):
            print(f"⚠️ Warning: {group_path} does not exist.")
            return

        user_count = 0  # Initialize a counter for processed users

        for user in os.listdir(group_path):
            if user.startswith('.') or user in ["Thumbs.db", "desktop.ini"]:
                continue  # Skip hidden/system files
            
            user_folder = os.path.join(group_path, user)
            if os.path.exists(user_folder):
                for file in os.listdir(user_folder):
                    if file.startswith('.'):
                        continue  # Skip hidden files

                    file_path = os.path.join(user_folder, file)
                    if file.endswith('Tweets.csv'):
                        self.process_csv(file_path, label)
                        user_count += 1

                # Stop if user limit is reached
                if user_limit is not None and user_count >= user_limit:
                    print(f"Processed {user_count} users; reached the specified limit.")
                    return

    def process_csv(self, file_path, label):
        try:
            df = pd.read_csv(file_path)
            if 'full_text' not in df.columns:
                raise ValueError("Column 'full_text' not found in the CSV file.")

            for text in df['full_text'].dropna():
                preprocessed_text = self.preprocess_text(str(text))
                if len(preprocessed_text.strip()) > 0 and not text.startswith('RT '):
                    tokenized_text = preprocessed_text.split()[:self.max_tokens]  # Tokenize by splitting on spaces
                    self.data.append((" ".join(tokenized_text), label))  # Store as text and label pair

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def preprocess_text(self, text):
        """ Cleans text: lowercasing, removing URLs, mentions, and emojis. """
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove user mentions
        text = re.sub(r"[^a-z.!? ]+", "", text)  # Remove non-alphabetic characters except ., !, ?
        
        # Remove emojis
        emoji_pattern = re.compile("["       
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # Chinese characters
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\ufe0f"
                               "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]

        # ✅ Tokenize text before returning it
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=384,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

if __name__ == "__main__":
    print("Initializing TwitterTextDataset...")
    dataset = TwitterTextDataset("../data")
    print("Dataset initialized successfully.")
    print(f"Total texts in dataset: {len(dataset)}")
