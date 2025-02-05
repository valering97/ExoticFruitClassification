import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from typing import Tuple, Dict

from pandas import DataFrame
from numpy import mean, std
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from src.constants import RANDOM_SEED


def plot_accuracy_vs_k(Ks, accuracies_train, accuracies_test):
    """
    Plot accuracy vs K for KNN model (Train vs Test).
    
    Parameters:
    Ks (list): List of K values used for the KNN classifier.
    accuracies_train (list)
    accuracies_test (list)
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the training accuracy
    plt.plot(Ks, accuracies_train, marker='o', linestyle='-', color='b', label="Train Accuracy")
    
    # Plot the test accuracy
    plt.plot(Ks, accuracies_test, marker='o', linestyle='-', color='g', label="Test Accuracy")
    
    plt.title("Accuracy vs K for KNN Classifier (Train vs Test)", fontsize=14)
    plt.xlabel("Number of Neighbors (K)", fontsize=12)
    plt.ylabel("Accuracy Score", fontsize=12)
    plt.grid(True)
    plt.xticks(Ks) 
    plt.legend()
    
    plt.show()


def cross_validation_model(df: DataFrame, target: str, n_neighbors: int, cv: int = 5) -> DataFrame:
    """
    Perform cross-validation for a KNN model with a specified number of neighbors.

    Parameters:
    df (DataFrame): DataFrame containing features and the target column.
    target (str): The name of the target column in the DataFrame.
    n_neighbors (int): The number of neighbors to use for the KNN classifier.
    cv (int): The number of folds for cross-validation (default is 5).

    Returns:
    DataFrame: DataFrame with cross-validation metrics for each fold.
    """
    X = df.drop(columns=[target])
    y = df[target]

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

    metrics: Dict[str, list] = {
        'fold': [],
        'train_accuracy': [], 'test_accuracy': [],
        'train_precision': [], 'test_precision': [],
        'train_recall': [], 'test_recall': [],
        'train_f1': [], 'test_f1': [],
        'train_loss': [], 'test_loss': []
    }

    print("=" * 40)
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print(f"Training KNN in Fold {fold + 1}...")
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)

        y_train_pred = knn.predict(X_train)
        y_test_pred = knn.predict(X_test)
        y_train_proba = knn.predict_proba(X_train)
        y_test_proba = knn.predict_proba(X_test)

        metrics['fold'].append(fold + 1)
        metrics['train_accuracy'].append(accuracy_score(y_train, y_train_pred))
        metrics['test_accuracy'].append(accuracy_score(y_test, y_test_pred))
        metrics['train_precision'].append(precision_score(y_train, y_train_pred, average='weighted'))
        metrics['test_precision'].append(precision_score(y_test, y_test_pred, average='weighted'))
        metrics['train_recall'].append(recall_score(y_train, y_train_pred, average='weighted'))
        metrics['test_recall'].append(recall_score(y_test, y_test_pred, average='weighted'))
        metrics['train_f1'].append(f1_score(y_train, y_train_pred, average='weighted'))
        metrics['test_f1'].append(f1_score(y_test, y_test_pred, average='weighted'))
        metrics['train_loss'].append(log_loss(y_train, y_train_proba))
        metrics['test_loss'].append(log_loss(y_test, y_test_proba))

    print("=" * 40)
    results_cv = DataFrame(metrics)

    return results_cv
