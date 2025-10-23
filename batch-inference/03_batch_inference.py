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

!pip3 install open-inference-openapi
!pip3 install --upgrade cdpcli

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Libraries imported successfully!")

def get_krb_jwt_token():
    try:
        with open("/tmp/jwt", "r") as jwt_file:
            jwt_data = json.load(jwt_file)
            return jwt_data["access_token"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error: {e}")
        return None
print("Function to get kerberos JWT defined!")

# Demo configuration (replace with actual values for real inference)
DEMO_BASE_URL = BASE_URL
DEMO_MODEL_NAME = f"{MODEL_ID}"

# If you are running on-prem, please use get_ums_jwt_token() instead.
CDP_TOKEN = get_krb_jwt_token()

# Initialize demo inference client
demo_client = TritonBatchInference(DEMO_BASE_URL, DEMO_MODEL_NAME, CDP_TOKEN)

print("Demo inference client initialized")
print(f"Base URL: {DEMO_BASE_URL}")
print(f"Model Name: {DEMO_MODEL_NAME}")

# Run demo inference pipeline
print("=" * 50)
print("DEMO BATCH INFERENCE PIPELINE")
print("=" * 50)

# Check server status (demo)
if demo_client.check_server_status():
    print("✓ Demo server ready")

# Get model configuration (demo)
config = demo_client.get_triton_model_config()
batch_size = config['preferred_batch_size'][0] if config else 8
print(f"Using batch size: {batch_size}")

# Prepare data
X_inference, y_inference, class_names = demo_client.prepare_iris_data()

# Run demo inference
print("\nRunning demo batch inference...")
start_time = time.time()
demo_predictions = demo_client.run_batch_inference_demo(X_inference, batch_size)
total_time = time.time() - start_time

# Evaluate demo results
results = demo_client.evaluate_predictions(demo_predictions, y_inference, class_names)
results['total_inference_time'] = total_time
results['avg_time_per_sample'] = total_time / len(X_inference)
results['throughput_samples_per_second'] = len(X_inference) / total_time

print("\n" + "=" * 50)
print("DEMO INFERENCE RESULTS")
print("=" * 50)
print(f"Total Samples: {results['total_samples']}")
print(f"Correct Predictions: {results['correct_predictions']}")
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Total Inference Time: {results['total_inference_time']:.3f}s")
print(f"Average Time per Sample: {results['avg_time_per_sample']:.6f}s")
print(f"Throughput: {results['throughput_samples_per_second']:.2f} samples/second")

print("\nDemo Classification Report:")
print(results['classification_report'])
