import librosa
import numpy as np

def extract_features_all_types(file_path):
    """
    Extracts multiple types of audio features from an audio file.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        np.array: A combined array of extracted features including MFCCs, Chroma, Spectral Contrast,
                  Zero-Crossing Rate, Root Mean Square Energy, and Mel-Spectrogram.
    """
    # Load audio file
    y, sr = librosa.load(file_path, sr=32000)

    # Extract MFCCs (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Extract Chroma feature
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Extract Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

    # Extract Zero-Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)

    # Extract Root Mean Square Energy
    rmse = librosa.feature.rms(y=y)
    rmse_mean = np.mean(rmse)

    # Extract Mel-Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_mean = np.mean(mel_spectrogram, axis=1)

    # Combine all features into a single array
    features = np.hstack((mfccs_mean, chroma_mean, spectral_contrast_mean,
                          zero_crossing_rate_mean, rmse_mean, mel_spectrogram_mean))

    return features

def extract_features_mfcc(file_path):
    """
    Extracts MFCC (Mel-frequency cepstral coefficients) features from an audio file.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        np.array: An array of mean MFCC features.
    """
    # Load audio file
    y, sr = librosa.load(file_path, sr=32000)

    # Extract MFCCs (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)

    # Return mean MFCC features
    return np.mean(mfccs, axis=1)
