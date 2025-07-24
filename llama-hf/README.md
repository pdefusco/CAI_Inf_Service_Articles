# Deploy Llama3b-Instruct to AI Inference Service from HF Catalog Programmatically

## Objective

In this tutorial you will learn how to programmatically deploy Meta's Llama-3.1-8B-Instruct LLM to the Cloudera AI Inference Service. First, you will download the model to the Cloudera AI Registry; Then, you will create an AI Inference Service Endpoint to serve predictions in real time from within your cluster.

### Cloudera AI & LLM's

Cloudera AI (CAI) is a platform that enables organizations to build, train, and deploy machine learning and artificial intelligence models at scale. One of its key features is the Cloudera AI Inference Service, which allows users to easily deploy large language models (LLMs) for real-time or batch inference. With Cloudera AI, data scientists and engineers can manage and serve LLMs like Llama, Mistral, or open-source GPT models using containerized environments and scalable infrastructure. This service supports secure, low-latency model serving, making it easier to integrate AI into enterprise applications.

### Hybrid Enterprise AI with CAI

Cloudera AI (CAI) is a core component of Cloudera’s hybrid cloud data platform, which is designed to operate seamlessly across both private and public cloud environments. This hybrid architecture allows organizations to deploy AI models securely wherever their data resides—on-premises for sensitive workloads or in the public cloud for greater scalability and flexibility. With Cloudera AI, enterprises can maintain governance, compliance, and control over their machine learning pipelines while taking advantage of cloud-native capabilities. This ensures that large language models and other AI applications can be deployed and managed securely across diverse IT environments without compromising performance or data privacy.

### CAI Integration with the Hugging Face Catalog

CAI simplifies access to cutting-edge machine learning models through its integration with the Hugging Face Model Catalog, a popular repository for open-source models, including large language models (LLMs). This integration enables users to easily browse, select, and download pre-trained models directly from the Hugging Face Catalog into their Cloudera environment. Whether it's BERT, GPT, Llama, or other advanced models, Cloudera AI streamlines the process of importing and deploying them for inference or fine-tuning. This seamless connection accelerates experimentation and development while ensuring that models can be securely managed and deployed within Cloudera’s governed data platform.

## Requirements

This example was built with Private Cloud 1.5.5 and CAI 2.0.49 but it will also work in Public Cloud without any changes. You can reproduce this tutorial in your CAI environment with the following:

* A HuggingFace Account and Token.
* A local installation of the CDP CLI.
* A CAI Environment in Private or Public Cloud.
* An AI Registry deployment.
* An AI Inference Service deployment.

The Private Cloud environment used for this example is not airgapped. For help with an airgapped environment please use the documentation at this link.

## Useful Documentation Links

* How to deploy a Workbench in Cloudera AI.
* How to deploy an AI Registry in Cloudera AI.
* How to deploy an AI Inference Service in Cloudera AI.
* How to set up a Hugging Face Account.
* How to set up the CDP CLI.

### Tutorial

All artifacts are included in this Git repository.

#### 1. Clone the Git Repository as a CAI Project

#### 2. Generate a CDP Token

#### 3. Create the Project Environment Variables with Secrets

#### 4. Launch a CAI Session and Run the Script to Download the Model Programmatically

#### 5. Monitor Model Download from the AI Registry UI

#### 6. Deploy the Model Programmatically

## Summary & Next Steps

In
