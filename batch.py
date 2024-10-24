import numpy as np
import pandas as pd
import os
import json
import torch

class BatchLoader:
    def __init__(self, keypoint_dir, csv_file, batch_size, vocabulary, fps=30):
        """
        Initialize the batch loader with keypoint directory, CSV file, batch size, and other parameters.
        
        keypoint_dir: Directory containing OpenPose keypoints in JSON format.
        csv_file: Path to the CSV file containing text data aligned with keypoints.
        batch_size: Number of samples per batch.
        vocabulary: A Vocabulary object to tokenize text data.
        fps: Frames per second for time alignment.
        """
        self.keypoint_dir = keypoint_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.fps = fps
        
        # Load CSV containing text alignments and sentence-level information
        try:
            self.data = pd.read_csv(csv_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {csv_file}")
        
        # Shuffle the data for randomness in training
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        
        # Get the list of available samples and reset index
        self.index = 0

    def load_keypoints(self, sentence_id):
        """
        Load keypoints for a given sentence from the JSON directory.
        Sentence ID refers to the folder containing individual frame JSON files.
        
        Returns:
        - keypoints as a torch.Tensor, or None if keypoints could not be loaded.
        """
        keypoint_dir = os.path.join(self.keypoint_dir, sentence_id)
        if not os.path.exists(keypoint_dir):
            print(f"Warning: Keypoint directory does not exist: {keypoint_dir}")
            return None

        keypoint_files = sorted(os.listdir(keypoint_dir))
        all_keypoints = []

        # Loop through each frame JSON file
        for file in keypoint_files:
            try:
                with open(os.path.join(keypoint_dir, file), 'r') as f:
                    data = json.load(f)
                    if len(data["people"]) > 0:
                        # Extract keypoints for the first detected person
                        keypoints = np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)
                        all_keypoints.append(keypoints)
                    else:
                        # Handle case where no person is detected
                        print(f"Warning: No people detected in file: {file}")
            except json.JSONDecodeError:
                print(f"Warning: Malformed JSON file: {file}")
            except KeyError:
                print(f"Warning: Unexpected JSON structure in file: {file}")

        if len(all_keypoints) == 0:
            return None  # Return None if no valid keypoints were loaded

        # Ensure consistent tensor creation
        return torch.tensor(all_keypoints)

    def tokenize_text(self, text):
        """
        Tokenize the input text using the provided vocabulary and return token ids.
        
        text: A string representing the sentence.
        Returns: A tensor of token IDs.
        """
        tokens = self.vocabulary.tokenize(text)
        token_ids = self.vocabulary.convert_tokens_to_ids(tokens)
        return torch.tensor(token_ids)

    def collate_fn(self, batch):
        """
        Collation function to handle padding for variable-length sequences of both text and keypoints.
        This is used to ensure all sequences in the batch are of uniform length.
        
        Input:
        - batch: List of (text, keypoints) tuples.
        
        Returns:
        - Padded text tensor, padded keypoints tensor.
        """
        # Unzip the batch into texts and keypoints
        texts, keypoints = zip(*batch)
        
        # Pad text sequences to the length of the longest sequence in the batch
        text_lengths = [len(text) for text in texts]
        max_text_len = max(text_lengths)
        padded_texts = torch.zeros((len(texts), max_text_len), dtype=torch.long)
        
        for i, text in enumerate(texts):
            padded_texts[i, :len(text)] = text

        # Validate and pad keypoint sequences
        keypoint_lengths = [len(k) for k in keypoints if k is not None]
        if len(keypoint_lengths) == 0:
            raise ValueError("All keypoint sequences are empty or invalid in the batch.")

        max_kp_len = max(keypoint_lengths)
        kp_dim = keypoints[0].size(1) if keypoints[0] is not None else None

        padded_keypoints = torch.zeros((len(keypoints), max_kp_len, kp_dim)) if kp_dim else None

        for i, kp in enumerate(keypoints):
            if kp is not None and kp.size(1) == kp_dim:
                padded_keypoints[i, :len(kp)] = kp
            else:
                print(f"Warning: Keypoint sequence dimensions do not match for batch item {i}")

        return padded_texts, padded_keypoints

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.data)

    def __iter__(self):
        """
        Create an iterator that returns batches of samples during training.
        """
        self.index = 0  # Reset index when iteration begins
        
        while self.index < len(self.data):
            batch = []
            
            # Get batch of samples
            for _ in range(self.batch_size):
                if self.index >= len(self.data):
                    break
                
                # Fetch row corresponding to the current sample
                row = self.data.iloc[self.index]
                sentence_id = row['video_id']  # This assumes video_id corresponds to the keypoint directory
                
                # Load keypoints and text
                keypoints = self.load_keypoints(sentence_id)
                if keypoints is None:
                    self.index += 1
                    continue  # Skip this sample if keypoints are invalid
                
                text = self.tokenize_text(row['sentence'])
                
                # Add to batch
                batch.append((text, keypoints))
                
                self.index += 1
            
            if len(batch) > 0:
                yield self.collate_fn(batch)  # Return the collated batch

        self.index = 0  # Reset index at the end of the iteration
