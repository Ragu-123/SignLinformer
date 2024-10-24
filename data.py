import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SignLanguageDataset(Dataset):
    def __init__(self, csv_file, keypoint_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            keypoint_dir (string): Directory with all the keypoint JSON files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.keypoint_dir = keypoint_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            A tuple (keypoints, sentence, start_time, end_time) for each sentence in the dataset.
        """
        try:
            row = self.data.iloc[idx]
            sentence_name = row['SENTENCE_NAME']
            sentence = row['SENTENCE']
            start_time = row['START_REALIGNED']
            end_time = row['END_REALIGNED']

            keypoint_folder = os.path.join(self.keypoint_dir, sentence_name)
            keypoints = self._load_keypoints(keypoint_folder, start_time, end_time)

            if self.transform:
                keypoints = self.transform(keypoints)

            return keypoints, sentence, start_time, end_time

        except Exception as e:
            print(f"Error loading data for index {idx}: {e}")
            return None

    def _load_keypoints(self, folder_path, start_time, end_time):
        """
        Load keypoints for a sentence from frame-by-frame JSON files.

        Args:
            folder_path (string): Path to the folder containing keypoint JSON files.
            start_time (float): Start time of the sentence.
            end_time (float): End time of the sentence.

        Returns:
            torch.Tensor: Tensor of shape (T, num_keypoints, 3) where T is the number of frames.
        """
        try:
            json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])

            # Estimate the frame range from the start/end time
            fps = 30  # Assuming 30fps
            start_frame = int(fps * start_time)
            end_frame = int(fps * end_time)

            keypoints_list = []
            for i, file_name in enumerate(json_files[start_frame:end_frame]):
                json_path = os.path.join(folder_path, file_name)
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Handle case where no people are detected in the frame
                if len(data["people"]) == 0:
                    keypoints = np.zeros((25, 3))  # Assuming 25 body keypoints
                else:
                    keypoints = np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)

                keypoints_list.append(keypoints)

            # Convert to Tensor
            keypoints_tensor = torch.tensor(keypoints_list, dtype=torch.float32)

            return keypoints_tensor

        except Exception as e:
            print(f"Error loading keypoints from {folder_path}: {e}")
            return torch.zeros((1, 25, 3))  # Return a zero tensor as fallback


def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable-length keypoints.
    Args:
        batch: List of tuples (keypoints, sentence, start_time, end_time)

    Returns:
        Padded keypoints tensor, sentences, start times, and end times.
    """
    keypoints, sentences, start_times, end_times = zip(*batch)
    
    # Validate that all keypoints have the same number of dimensions
    try:
        max_len = max([k.size(0) for k in keypoints])
        padded_keypoints = torch.zeros((len(keypoints), max_len, keypoints[0].size(1), keypoints[0].size(2)))

        for i, k in enumerate(keypoints):
            padded_keypoints[i, :k.size(0), :, :] = k

        return padded_keypoints, sentences, start_times, end_times

    except Exception as e:
        print(f"Error during collation: {e}")
        return None

# Usage example:
if __name__ == "__main__":
    # Directories and files
    train_keypoints = "C:/Users/SEC/Downloads/miniproject/dataset/train/openpose_output/json"
    train_csv = "C:/Users/SEC/Downloads/miniproject/dataset/How2Sign/sentence_level/train/text/en/raw_text/re_aligned/how2sign_realigned_train.csv"
    
    # Initialize dataset
    dataset = SignLanguageDataset(csv_file=train_csv, keypoint_dir=train_keypoints)

    # DataLoader with custom collate function
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Example of iterating through batches
    for batch in dataloader:
        keypoints, sentences, start_times, end_times = batch
        print(f"Batch of keypoints: {keypoints.shape}")
