import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import load_data
from visualization import plot_correlation_heatmap, plot_cm
from train_model import train_random_forest, train_cnn_model
from predict import predict_label
from tensorflow.keras import models

if __name__ == "__main__":
    dataset_path = '../dataset'
    data, labels, labels_name = load_data(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

    # if you wanna see correlation of faetures(If you dont comment below line)
    plot_correlation_heatmap(data)

    # if you wanna train Random Forest from Machine Learning Methods(If you dont comment below line)
    rf_model, y_pred = train_random_forest(X_train, X_test, y_train)
    plot_cm(y_test, y_pred, labels_name)

    X_train = X_train.reshape(-1, X_train.shape[1], 1)
    X_test = X_test.reshape(-1, X_test.shape[1], 1)
    # If you wanna train data on CNN Model (if you dont comment this below code)
    cnn_model, y_pred = train_cnn_model(X_train, y_train, X_test)
    plot_cm(y_test,y_pred, labels_name)
    cnn_model.save('audio_classification_model.h5')

    # load trained CNN model from directory and pass it to model for get Label
    loaded_model = models.load_model('audio_classification_model.h5')

    audio_path = '/home/reza/code/deeplearning/paya_task/dataset/telephony/telephony_31.wav'
    label = predict_label(audio_path, loaded_model, labels_name)
    print(f'The predicted label for the audio file is: {label}')
