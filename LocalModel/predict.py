import tensorflow as tf
from feature_extraction import extract_features_mfcc

def predict_label(file_path, model, labels_name):
    """
    Predicts the label for an audio file using a trained model.

    Args:
        file_path (str): Path to the audio file to be predicted.
        model (tf.keras.Model): Trained model used for prediction.
        labels_name (list): List of label names where index corresponds to the label.

    Returns:
        str: Predicted label name.
    """
    # Extract MFCC features from the audio file and reshape for model input
    features = extract_features_mfcc(file_path).reshape(1, -1, 1)

    # Predict the label using the trained model
    prediction = model.predict(features)

    # Convert the prediction probability to binary label
    label = int(prediction[0, 0] > 0.5)

    # Return the label name corresponding to the predicted label
    return labels_name[label]

