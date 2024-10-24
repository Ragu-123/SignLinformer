import os
import pandas as pd
import re
from collections import Counter
from typing import List, Dict

class Vocabulary:
    def __init__(self, pad_token: str = "<PAD>", sos_token: str = "<SOS>", eos_token: str = "<EOS>", unk_token: str = "<UNK>"):
        """
        Initializes the Vocabulary with special tokens.
        """
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        
        # Initialize with special tokens
        self.add_special_tokens()

    def add_special_tokens(self):
        """Adds special tokens to the vocabulary."""
        special_tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        for token in special_tokens:
            self.add_word(token)

    def add_word(self, word: str):
        """Adds a word to the vocabulary if it's not already present."""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.word_count[word] += 1  # Track frequency of the word for analysis
    
    def tokenize_sentence(self, sentence: str) -> List[str]:
        """Tokenizes a sentence into words/tokens, removing punctuation and converting to lowercase."""
        # Convert to lowercase and remove punctuation
        sentence = sentence.lower()
        tokens = re.findall(r'\b\w+\b', sentence)  # Extract words (alphanumeric)
        return tokens

    def build_vocabulary(self, csv_paths: List[str], min_freq: int = 1):
        """
        Builds the vocabulary from a list of CSV files containing the sentences.
        Args:
        - csv_paths: List of paths to CSV files.
        - min_freq: Minimum frequency a word must have to be included in the vocabulary.
        """
        for csv_path in csv_paths:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file {csv_path} does not exist.")
            
            # Load CSV data
            df = pd.read_csv(csv_path)
            if "SENTENCE" not in df.columns:
                raise ValueError(f"Expected 'SENTENCE' column in CSV {csv_path}, but not found.")

            # Tokenize sentences and update vocabulary
            for sentence in df["SENTENCE"].dropna():
                tokens = self.tokenize_sentence(sentence)
                for token in tokens:
                    self.word_count[token] += 1

        # Filter vocabulary by minimum frequency
        for word, count in self.word_count.items():
            if count >= min_freq:
                self.add_word(word)
    
    def sentence_to_indices(self, sentence: str) -> List[int]:
        """Converts a sentence into a list of token indices."""
        tokens = self.tokenize_sentence(sentence)
        return [self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens]
    
    def indices_to_sentence(self, indices: List[int]) -> str:
        """Converts a list of token indices back into a sentence."""
        return ' '.join([self.idx2word.get(idx, self.unk_token) for idx in indices])

    def save_vocabulary(self, path: str):
        """Saves the vocabulary to a JSON file."""
        vocab_data = {
            "word2idx": self.word2idx,
            "idx2word": self.idx2word
        }
        pd.DataFrame(vocab_data).to_json(path, orient='columns')

    def load_vocabulary(self, path: str):
        """Loads the vocabulary from a JSON file."""
        vocab_data = pd.read_json(path, orient='columns')
        self.word2idx = vocab_data["word2idx"].to_dict()
        self.idx2word = vocab_data["idx2word"].to_dict()

# Example Usage:
if __name__ == "__main__":
    csv_train = r"C:\Users\SEC\Downloads\miniproject\dataset\How2Sign\sentence_level\train\text\en\raw_text\re_aligned\how2sign_realigned_train.csv"
    csv_val = r"C:\Users\SEC\Downloads\miniproject\dataset\How2Sign\sentence_level\val\text\en\raw_text\re_aligned\how2sign_realigned_val.csv"
    
    vocab = Vocabulary()
    
    # Build vocabulary from train and validation CSVs
    vocab.build_vocabulary([csv_train, csv_val], min_freq=2)
    
    # Example: Converting a sentence to indices
    example_sentence = "life is beautiful"
    indices = vocab.sentence_to_indices(example_sentence)
    print(f"Sentence: {example_sentence} -> Indices: {indices}")
    
    # Save the vocabulary
    vocab.save_vocabulary("vocab.json")
    
    # Load vocabulary (example)
    vocab.load_vocabulary("vocab.json")
