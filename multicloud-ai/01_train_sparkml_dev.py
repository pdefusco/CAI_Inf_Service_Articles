#****************************************************************************
# (C) Cloudera, Inc. 2020-2025
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

!pip install scikit-learn>=1.3.0 onnxmltools>=1.12.0 skl2onnx>=1.15.0 numpy>=1.23.0 onnxruntime==1.21.0

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

# Start Spark session
spark = SparkSession.builder.appName("SparkML_ONNX_MLflow").getOrCreate()

# -----------------------------------------
# 1. Create a small synthetic training dataset
# -----------------------------------------
data = [
    (1.0, 0.1, 0.2),
    (0.0, -0.3, 0.8),
    (1.0, 0.5, 0.4),
    (0.0, -0.2, -0.5),
    (1.0, 0.6, 0.1)
]

df = spark.createDataFrame(data, ["label", "f1", "f2"])

assembler = VectorAssembler(inputCols=["f1", "f2"], outputCol="features")
assembled = assembler.transform(df)

# -----------------------------------------
# 2. Train a Spark ML classifier
# -----------------------------------------

lr = LogisticRegression(featuresCol="features", labelCol="label")
model = lr.fit(assembled)

print("Coefficients:", model.coefficients)
print("Intercept:", model.intercept)

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLR
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Create sklearn logistic regression model with Spark parameters
sk_model = SklearnLR()
sk_model.coef_ = np.array([model.coefficients.toArray()])
sk_model.intercept_ = np.array([model.intercept])
sk_model.classes_ = np.array([0., 1.])   # binary classification

# Define ONNX input type: batch of 2 features
initial_type = [('input', FloatTensorType([None, 2]))]

# Convert to ONNX
onnx_model = convert_sklearn(sk_model, initial_types=initial_type)

# Save ONNX model locally
onnx_path = "spark_lr_model.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model saved:", onnx_path)

# -----------------------------------------
# 3. Log Onnx Model to AI Registry
# -----------------------------------------

import onnxruntime as ort

import mlflow

mlflow.set_experiment("spark-onnx-example")

with mlflow.start_run():

    # Log Spark model normally (optional)
    mlflow.spark.log_model(model, "spark-model")

    # Log the ONNX model artifact
    #mlflow.log_artifact(onnx_path, artifact_path="onnx-model")

    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_param("framework", "spark->sklearn->onnx")
    mlflow.onnx.log_model(
        onnx_model,
        "sparkml-model",
        registered_model_name=f"sparkml-clf"
    )

print("MLflow logging complete.")
