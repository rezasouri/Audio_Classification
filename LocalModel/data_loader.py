import os
import numpy as np
from feature_extraction import extract_features_mfcc

def load_data(dataset_path):
    """
    Loads audio data and their corresponding labels from a dataset directory.

    Args:
        dataset_path (str): Path to the dataset directory. The directory should contain
                            subdirectories for each label, named according to the labels.

    Returns:
        tuple: A tuple containing:
            - np.array: Array of extracted feature data.
            - np.array: Array of corresponding labels.
            - list: List of label names.
    """
    # Initialize lists to store data and labels
    datas, labels = [], []

    # Define the label names (subdirectory names)
    labels_name = ['microphony', 'telephony']

    # Iterate over each label
    for label in labels_name:
        folder_path = os.path.join(dataset_path, label)  # Path to the label's subdirectory

        # Iterate over each file in the subdirectory
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)  # Full path to the file

            # Extract MFCC features from the audio file
            features = extract_features_mfcc(file_path)

            # Append the features and corresponding label to the lists
            datas.append(features)
            labels.append(0 if label == 'microphony' else 1)

    # Convert lists to numpy arrays
    return np.array(datas), np.array(labels), labels_name
