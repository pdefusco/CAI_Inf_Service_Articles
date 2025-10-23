#****************************************************************************
# (C) Cloudera, Inc. 2020-2023
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

# Import required libraries
import os
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.pytorch
import mlflow.onnx
import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns

# Import inference-related libraries
from open_inference.openapi.client import OpenInferenceClient, InferenceRequest
import httpx
import json
import time
from urllib.parse import urlparse, urlunparse
from typing import Optional, Dict, Any, List

# Used while configuring CDP credentials config
import getpass

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Libraries imported successfully!")


class IrisClassifier(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, num_classes=3):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

print("Model architecture defined!")

def prepare_data():
    """Load and prepare the iris dataset"""
    iris = load_iris()
    X, y = iris.data, iris.target

    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {iris.target_names}")
    print(f"Features: {iris.feature_names}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler, iris

# Prepare the data
X_train, X_test, y_train, y_test, scaler, iris_data = prepare_data()

# Create a DataFrame for visualization
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df['target'] = iris_data.target
iris_df['species'] = iris_df['target'].map(dict(enumerate(iris_data.target_names)))

# Create pairplot
plt.figure(figsize=(12, 8))
sns.pairplot(iris_df, hue='species', height=2.5)
plt.suptitle('Iris Dataset - Feature Relationships', y=1.02)
plt.show()

# Class distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=iris_df, x='species')
plt.title('Class Distribution')
plt.show()

def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    """Train the model and return training history"""
    model.train()
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return train_losses, train_accuracies

def evaluate_model(model, test_loader):
    """Evaluate the model on test data"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    return accuracy, all_predictions, all_targets

print("Training functions defined!")

# Set MLflow experiment
mlflow.set_experiment("iris_classification_notebook")

# Hyperparameters
hidden_size = 16
learning_rate = 0.01
num_epochs = 100
batch_size = 16

print(f"Training configuration:")
print(f"Hidden size: {hidden_size}")
print(f"Learning rate: {learning_rate}")
print(f"Number of epochs: {num_epochs}")
print(f"Batch size: {batch_size}")

with mlflow.start_run() as run:
    print("Starting MLflow experiment...")

    # Log hyperparameters
    mlflow.log_param("hidden_size", hidden_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("batch_size", batch_size)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = IrisClassifier(input_size=4, hidden_size=hidden_size, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\nTraining model...")
    # Train the model
    train_losses, train_accuracies = train_model(
        model, train_loader, criterion, optimizer, num_epochs
    )

    # Evaluate the model
    print("\nEvaluating model...")
    test_accuracy, predictions, targets = evaluate_model(model, test_loader)

    # Log metrics
    mlflow.log_metric("final_train_accuracy", train_accuracies[-1])
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("final_train_loss", train_losses[-1])

    # Log training curves
    for epoch, (loss, acc) in enumerate(zip(train_losses, train_accuracies)):
        mlflow.log_metric("train_loss", loss, step=epoch)
        mlflow.log_metric("train_accuracy", acc, step=epoch)

    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
    print(f"MLflow Run ID: {run.info.run_id}")

    # Store the run_id for later use
    training_run_id = run.info.run_id

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss curve
ax1.plot(train_losses)
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)

# Accuracy curve
ax2.plot(train_accuracies)
ax2.set_title('Training Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Create and display classification report
class_names = iris_data.target_names
report = classification_report(targets, predictions, target_names=class_names)
print("Classification Report:")
print(report)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(targets, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

def convert_to_onnx(model, input_size=(1, 4), onnx_path="iris_model.onnx"):
    """Convert PyTorch model to ONNX format"""
    model.eval()

    # Create dummy input for tracing
    dummy_input = torch.randn(input_size)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model exported to ONNX format: {onnx_path}")
    return onnx_path

def verify_onnx_model(onnx_path, test_data):
    """Verify ONNX model works correctly"""
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)

    # Test with a small batch
    test_input = test_data[:5].numpy()  # Take first 5 samples
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_outputs = ort_session.run(None, ort_inputs)

    print(f"ONNX model verification successful. Output shape: {ort_outputs[0].shape}")
    return True

# Convert and verify ONNX model
onnx_path = "iris_model.onnx"
convert_to_onnx(model, input_size=(1, 4), onnx_path=onnx_path)
verify_onnx_model(onnx_path, X_test)

REGISTERED_MODEL_NAME_ONNX = "iris_onnx_classifier_notebook"
with mlflow.start_run() as run:
    # Log the PyTorch model to MLflow with input example and signature
    print("Logging PyTorch model to MLflow...")
    mlflow.pytorch.log_model(
        model,
        "iris_classifier_pytorch",
        registered_model_name="iris_pytorch_classifier_notebook",
        input_example=input_example,
        signature=signature
    )

    # Log the ONNX model to MLflow with input example and signature
    print("Logging ONNX model to MLflow...")
    onnx_model = onnx.load(onnx_path)
    mlflow.onnx.log_model(
        onnx_model,
        "iris_classifier_onnx",
        registered_model_name=f"{REGISTERED_MODEL_NAME_ONNX}",
        input_example=input_example,
        signature=signature
    )

    print("Models logged successfully to MLflow!")
    print(f"MLflow Run ID: {run.info.run_id}")
    print("✓ Models logged with input examples and signatures!")
