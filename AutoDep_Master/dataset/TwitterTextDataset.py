import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import re
from transformers import AlbertTokenizer

class TwitterTextDataset(Dataset):
    def __init__(self, root_dir, tokenizer=None, max_tokens=512):
        # Get the absolute path of the directory where this script resides
        script_dir = os.getcwd()
        
        # Construct the absolute path to the dataset root directory
        self.root_dir = os.path.abspath(os.path.join(script_dir, root_dir))
        print(self.root_dir)
        self.data = []

        self.tokenizer = tokenizer if tokenizer else AlbertTokenizer.from_pretrained("albert-base-v2")
        self.max_tokens = max_tokens

        # Load control group texts
        control_path = os.path.join(self.root_dir, "control_group", "users")
        print(f"Loading control group from: {control_path}")
        self.load_texts(control_path, label=0, user_limit = 10)

        # Load diagnosed group texts
        diagnosed_path = os.path.join(self.root_dir, "diagnosed_group", "users")
        print(f"Loading diagnosed group from: {diagnosed_path}")
        self.load_texts(diagnosed_path, label=1, user_limit = 10)

        # Shuffle data for randomness
        random.shuffle(self.data)

        print(f"Dataset successfully loaded with {len(self.data)} text entries.")

    def load_texts(self, group_path, label, user_limit=None):
        if not os.path.exists(group_path):
            print(f"Warning: {group_path} does not exist.")
            return

        user_count = 0  # Initialize a counter for processed users
        print("User Limit:", user_limit)

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
                    if file.endswith('Tweets.csv'):
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
            if 'full_text' not in df.columns:
                raise ValueError("Column 'full_text' not found in the CSV file.")
            user_tweets = []
            for text in df['full_text'].dropna():
                preprocessed_text = self.preprocess_text(str(text))
                if len(preprocessed_text.strip()) > 0 and not text.startswith('RT '):
                    user_tweets.append(preprocessed_text)
            tokenized_tweets = self.tokenizer(user_tweets, truncation=False)["input_ids"]  # Tokenize all tweets

            # Chunking
            # We did chunking because Bert accepts max 512 token
            chunk = []
            chunk_size = 0
            for tweet_tokens in tokenized_tweets:
                if chunk_size + len(tweet_tokens) > self.max_tokens:
                    self.data.append((" ".join(self.tokenizer.batch_decode(chunk, skip_special_tokens=True)), label))
                    chunk = [tweet_tokens]
                    chunk_size = len(tweet_tokens)  # Reset chunk size to current tweet size
                else:
                    chunk.append(tweet_tokens)
                    chunk_size += len(tweet_tokens)

            
            if chunk:
                self.data.append((" ".join(self.tokenizer.batch_decode(chunk, skip_special_tokens=True)), label))
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        # Remove all punctuation except ., !, ?
        text = re.sub(r"[^a-z.!? ]+", "", text)
        
        # Remove emojis
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]

        # Tokenize text
        encoded = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_tokens, 
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

if __name__ == "__main__":
    print("Initializing TwitterTextDataset...")
    dataset = TwitterTextDataset("../data")
    print("Dataset initialized successfully.")
    print(f"Total texts in dataset: {len(dataset)}")