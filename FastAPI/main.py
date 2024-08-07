import numpy as np
import tensorflow as tf
import logging
from fastapi import FastAPI, File, UploadFile
from predict import predict_label

# Configure the logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model('models/audio_classification_model.h5')
labels_name = ['microphony', 'telephony']

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    """
    Endpoint to predict the label of an uploaded audio file.

    Args:
        file (UploadFile): Audio file to be predicted.

    Returns:
        dict: A dictionary containing the predicted label or an error message.
    """
    try:
        # Predict label of the uploaded audio file
        label = predict_label(file.file, model, labels_name=labels_name)

        # Log the prediction
        logger.info(f"Predicted label: {label}")

        return {"label": label}
    except Exception as e:
        # Log the error
        logger.error(f"Error occurred during prediction: {e}")
        return {"error": str(e)}
