import pandas as pd
import numpy as np
import pickle
import os
import mlflow
import mlflow.sklearn
import json
import argparse
import shutil

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import load_npz
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import estimator_html_repr

def load_processed_data(data_path):
    # Memuat data yang telah diproses
    print(f"Memuat data dari: {data_path}")
    # Load Sparse Matrix
    X_train = load_npz(os.path.join(data_path, "X_train.npz"))
    X_test = load_npz(os.path.join(data_path, "X_test.npz"))
    
    # Load Target
    with open(os.path.join(data_path, "y_train.pkl"), "rb") as f:
        y_train = pickle.load(f)
    with open(os.path.join(data_path, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)
        
    # Load Class Names
    with open(os.path.join(data_path, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    class_names = le.classes_
    
    return X_train, X_test, y_train, y_test, class_names

def plot_confusion_matrix(cm, class_names, title, filename):
    # Fungsi helper untuk membuat plot dan menyimpan confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix disimpan di {filename}")

# Fungsi utama yang baru
def main(alpha, loss, penalty, max_iter):
    """
    Fungsi training utama.
    """
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    with mlflow.start_run(run_name="CI Workflow Run") as run:
        print("Memulai run MLflow...")
        
        # 1. Log parameter yang diterima
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("loss", loss)
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("max_iter", max_iter)
        
        # 2. Muat data
        data_path = os.path.join(base_dir, "dataset-coursera_preprocessing")
        X_train, X_test, y_train, y_test, class_names = load_processed_data(data_path)
        
        # 3. Training model
        print("Melatih model SGDClassifier...")
        model = SGDClassifier(
            random_state=42,
            alpha=alpha,
            loss=loss,
            penalty=penalty,
            max_iter=max_iter
        )
        model.fit(X_train, y_train)
        print("Model selesai dilatih.")
        
        # 4. Evaluasi model (Sama seperti Kriteria 2)
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_pred_test, average='weighted')
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_f1 = f1_score(y_train, y_pred_train, average='weighted')
        
        # 5. Log metrik & artefak (Sama seperti Kriteria 2)
        print("Logging metrik dan artefak...")
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision_weighted", test_precision)
        mlflow.log_metric("test_recall_weighted", test_recall)
        mlflow.log_metric("test_f1_weighted", test_f1)
        mlflow.log_metric("training_accuracy", train_accuracy)
        mlflow.log_metric("training_f1_weighted", train_f1)
        
        # ... (Log metric_info.json) ...
        metrics_dict = {
            "test_accuracy": test_accuracy,
            "test_precision_weighted": test_precision,
            "test_recall_weighted": test_recall,
            "test_f1_weighted": test_f1,
            "training_accuracy": train_accuracy,
            "training_f1_weighted": train_f1
        }
        mlflow.log_dict(metrics_dict, "metric_info.json")

        # Log training_confusion_matrix.png
        cm_train = confusion_matrix(y_train, y_pred_train)
        plot_confusion_matrix(cm_train, class_names, 
                              title="Training Confusion Matrix", 
                              filename="training_confusion_matrix.png")
        mlflow.log_artifact("training_confusion_matrix.png")

        # Log estimator.html
        html_repr = estimator_html_repr(model)
        with open("estimator.html", "w", encoding="utf-8") as f:
            f.write(html_repr)
        mlflow.log_artifact("estimator.html")
        
        # Log model ke dagshub
        mlflow.sklearn.log_model(model, "model")

        # Simpan model lokal untuk docker build
        print("Menyimpan model secara lokal untuk Docker Build")
        local_model_path = os.path.join(base_dir, "local_model_for_docker")
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)
        
        mlflow.sklearn.save_model(model, local_model_path)
        print(f"Model disimpan secara lokal di '{local_model_path}'")
        print(f"Run {run.info.run_id} selesai dan dicatat")

# Entry point untuk menerima parameter
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--loss", type=str, required=True)
    parser.add_argument("--penalty", type=str, required=True)
    parser.add_argument("--max_iter", type=int, required=True)
    args = parser.parse_args()
    
    main(args.alpha, args.loss, args.penalty, args.max_iter)