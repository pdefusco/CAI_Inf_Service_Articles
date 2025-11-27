# CAI Inference Service Articles

Collection of examples with recommendations for deploying models to Cloudera AI Inference Service.

### Deploy Llama 3.1 8B Instruct to AI Inference Service from HF Catalog Programmatically

In this tutorial you will learn how to programmatically deploy Meta's Llama-3.1-8B-Instruct LLM to the Cloudera AI Inference Service. First, you will download the model to the Cloudera AI Registry; Then, you will create an AI Inference Service Endpoint to serve predictions in real time from within your Data Center.

If you'd like to use a different model you can apply the same steps for any language models available in the Hugging Face Catalog.

This example was built with Cloudera On Prem Private Cloud 1.5.5, CAI Workbench 2.0.49, Inference Service 1.4.0 and AI Registry 1.7.0. The same example will also work in Public Cloud without any changes.

Instructions & Code: https://github.com/pdefusco/CAI_Inf_Service_Articles/tree/main/llama-hf#deploy-llama-31-8b-instruct-to-ai-inference-service-from-hf-catalog-programmatically

### LLMOps Utils for Cloudera AI Inference Service

In this tutorial you will learn how to deploy Deepseek R1 Distill Llama 8B to the Cloudera AI Inference Service programmatically using Python and an LLMOps Util.

First, you will download the model to the Cloudera AI Registry; Then, you will create an AI Inference Service Endpoint to serve predictions in real time from within your Data Center.

The LLMOps util and the overall tutorial is particularly tailored for CAI users who want to complete the end to end lifecycle - from LLM download to Endpoint - entirely in Python. If you'd like to use a different model you can apply the same steps for any language models available in the NGC Catalog.  

This example was built with Cloudera On Cloud Public Cloud Runtime 7.3.1, CAI Workbench 2.0.50, Inference Service 1.4.0 and AI Registry 1.7.0. The same example will also work in Private Cloud with very minor changes.

Instructions & Code: https://github.com/pdefusco/CAI_Inf_Service_Articles/blob/main/llmops_utils/README.md#llmops-utils-for-cloudera-ai-inference-service

### Train, Register, Deploy and Serve XGBoost Classifier to AI Inference Service Programmatically

In this tutorial you will learn how to build an XGBoost classifier and deploy it to the Cloudera AI Inference Service. First, you will train and register the model with the Cloudera AI Registry; Then, you will create an AI Inference Service Endpoint to serve predictions in real time from your secure endpoint.

This example was built with Cloudera On Cloud Public Cloud Runtime 7.3.1, CAI Workbench 2.0.50, Inference Service 1.4.0 and AI Registry 1.7.0. The same example will also work in Private Cloud without changes.

Instructions & Code: https://github.com/pdefusco/CAI_Inf_Service_Articles/tree/main/xgboost#train-register-deploy-and-serve-xgboost-classifier-to-ai-inference-service-programmatically

### Deploy Mixtral 8x7B Instruct from NGC Catalog to AI Inference Service

In this tutorial you will learn how to programmatically deploy Mixtral 8x7B Instruct to the Cloudera AI Inference Service. First, you will download the model to the Cloudera AI Registry; Then, you will create an AI Inference Service Endpoint to serve predictions in real time from within your Public Cloud of choice.

If you'd like to use a different model you can apply the same steps for other language models available in the NGC Catalog.  

This example was built with Cloudera On Cloud Public Cloud 7.3.1, CAI Workbench 2.0.53, Inference Service 1.7.0 and AI Registry 1.11.0. The same example will also work in Private Cloud without any changes.

Instructions & Code: https://github.com/pdefusco/CAI_Inf_Service_Articles/tree/main/mixtral-ngc

### Hybrid AI: Train a PyTorch Model in Cloudera AI on AWS and Deploy it in Cloudera AI Inference Service OnPrem

In this tutorial you will learn how to programmatically train, register, and deploy a model with Cloudera AI Hybrid Cloud. First, you will train a PyTorch model in Cloudera AI on AWS. Then, you will register the model with Cloudera AI on premises. Finally, you will deploy the model to the Cloudera AI Inference Service, also running on premises.

The general purpose of the demo is to show an end to end hybrid AI workflow between Cloudera AI on AWS and on Prem. The same worfklow steps can be applied to other use cases, frameworks, and Cloudera AI on Azure.

This example was built with Cloudera on AWS and Cloudera OnPrem. At time of this writing, all latest component versions were used. In general, previus and/or future component versions will work as well.

Instructions & Code: https://github.com/pdefusco/CAI_Inf_Service_Articles/blob/main/hybrid-ai/README.md
