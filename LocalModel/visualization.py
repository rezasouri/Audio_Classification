import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def plot_correlation_heatmap(data):
    """
    Plots a heatmap of the correlation matrix for the given data.

    Args:
        data (array-like): Data for which to compute and plot the correlation matrix.
                           Expected shape (num_samples, num_features).
    """
    # Create feature names based on the number of columns in the data
    feature_names = [f'feature_{i}' for i in range(data.shape[1])]

    # Create a DataFrame from the data
    df = pd.DataFrame(data, columns=feature_names)

    # Compute the correlation matrix
    corr_matrix = df.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidths=0.5, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()

def plot_cm(y_test, y_pred, labels_name):
    """
    Plots a confusion matrix and prints a classification report.

    Args:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels_name (list): List of label names where index corresponds to the label.
    """
    # Print the classification report
    print(classification_report(y_test, y_pred))

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_name, yticklabels=labels_name)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
