import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType, Int64TensorType
import onnx

# Assume you have already trained a Spark ML pipeline model
# e.g.
# from pyspark.ml import PipelineModel
# spark_model = PipelineModel.load("path/to/my/spark_model")

spark_model = ...  # your loaded or trained PipelineModel

# Define the input schema/types: name + shape + dtype
# Suppose your model expects features vector of length N
N = 10  # number of features in your Spark feature vector
initial_types = [("features", FloatTensorType([None, N]))]

# Convert to ONNX
onnx_model = onnxmltools.convert_sparkml(
    spark_model,
    name="SparkMLPipelineModel",
    initial_types=initial_types,
    target_opset=None  # you can specify a target ONNX opset version if desired
)

# Save model to file
onnxmltools.utils.save_model(onnx_model, "spark_model.onnx")

print("ONNX opset version:", onnx_model.opset_import[0].version)



import mlflow
import mlflow.onnx
import onnx
import onnxruntime as ort
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# --- Train a simple sklearn model ---
X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# --- Convert to ONNX ---
initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# --- Start an MLflow run and log the model ---
mlflow.set_experiment("onnx-demo")

with mlflow.start_run(run_name="iris-onnx-example"):
    # log the ONNX model
    mlflow.onnx.log_model(
        onnx_model=onnx_model,
        artifact_path="model",
        registered_model_name="IrisClassifierONNX"  # optional, registers in MLflow Model Registry
    )

    # log some parameters/metrics
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", model.score(X, y))
