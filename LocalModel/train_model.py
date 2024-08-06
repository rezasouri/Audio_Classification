from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models

def train_random_forest(X_train, X_test, y_train):
    """
    Trains a Random Forest classifier on the given training data and returns the trained model and predictions on the test data.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Test data features.
        y_train (array-like): Training data labels.

    Returns:
        model (RandomForestClassifier): Trained Random Forest model.
        y_pred (array): Predictions on the test data.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, y_pred

def train_cnn_model(X_train, y_train, X_test):
    """
    Trains a Convolutional Neural Network (CNN) on the given training data and returns the trained model and predictions on the test data.

    Args:
        X_train (array-like): Training data features, expected shape (num_samples, num_features, 1).
        y_train (array-like): Training data labels.
        X_test (array-like): Test data features, expected shape (num_samples, num_features, 1).
        y_test (array-like): Test data labels.

    Returns:
        model (Sequential): Trained CNN model.
        y_pred (array): Predictions on the test data.
    """
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.7),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[])
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    return model, y_pred
