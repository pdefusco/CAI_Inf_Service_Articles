# Deploy Mixtral to AI Inference Service from NGC Catalog

![alt text](/img/step_1_model_download.png)

## Objective

In this tutorial you will learn how to programmatically deploy Mixtral to the Cloudera AI Inference Service. First, you will download the model to the Cloudera AI Registry; Then, you will create an AI Inference Service Endpoint to serve predictions in real time from within your Public Cloud of choice.

If you'd like to use a different model you can apply the same steps for other language models available in the NGC Catalog.  

### Motivation

With Cloudera AI, enterprises can download open source GenAI models and securely host them in their Public or Private Cloud, thus implementing their own LLM-powered applications while preventing proprietary information from being shared with closed-source model companies such as OpenAI.

### Cloudera AI & LLM's

Cloudera AI (CAI) is a platform that enables organizations to build, train, and deploy machine learning and artificial intelligence models at scale. One of its key features is the Cloudera AI Inference Service, which allows users to easily deploy large language models (LLMs) for real-time or batch inference. With Cloudera AI, data scientists and engineers can manage and serve LLMs like Llama, Mistral, or open-source GPT models using containerized environments and scalable infrastructure. This service supports secure, low-latency model serving, making it easier to integrate AI into enterprise applications.

### Hybrid Enterprise AI with CAI

Cloudera AI (CAI) is a core component of Cloudera’s hybrid cloud data platform, which is designed to operate seamlessly across both private and public cloud environments. This hybrid architecture allows organizations to deploy AI models securely wherever their data resides—on-premises for sensitive workloads or in the public cloud for greater scalability and flexibility. With Cloudera AI, enterprises can maintain governance, compliance, and control over their machine learning pipelines while taking advantage of cloud-native capabilities. This ensures that large language models and other AI applications can be deployed and managed securely across diverse IT environments without compromising performance or data privacy.

### CAI Integration with the NGC Catalog

NVIDIA NGC is a curated catalog and registry of GPU-optimized containers, pretrained models, SDKs, and full AI/HPC workflows from NVIDIA, designed to help users quickly build, train, and deploy AI workloads across cloud, on-premises, and edge environments.

Cloudera AI Registry (part of Cloudera’s model management offering) integrates with NGC so that models from the NGC catalog can be imported directly into the Registry—enabling enterprises to govern, deploy, and monitor NVIDIA-optimized models within their existing Cloudera AI workflows.

## Requirements

This example was built with Cloudera On Cloud Public Cloud 3.7.1, CAI Workbench 2.0.53, Inference Service 1.7.0 and AI Registry 1.10.0. The same example will also work in Private Cloud without any changes. You can reproduce this tutorial in your CAI environment with the following:

* A CAI Environment in Private or Public Cloud.
* An AI Registry deployment.
* An AI Inference Service deployment with g6e.12xlarge GPU node group / Autoscale Range Min Max 1-4 / Root Volume Size 512.
* A Python 3.11 Cloudera AI Runtime with PBJ Workbench IDE.

If you are operating in a Cloudera On Prem environment that is airgapped you have to follow a few extra steps to download the model with a python script shown as documented here: https://docs.cloudera.com/machine-learning/1.5.5/importing-model-airgapped/topics/ml-models-in-air-gapped-environment.html

## Useful Documentation Links

* How to deploy a Workbench in Cloudera AI: https://docs.cloudera.com/machine-learning/1.5.5/workspaces-privatecloud/topics/ml-pvc-provision-ml-workspace.html
* How to deploy an AI Registry in Cloudera AI: https://docs.cloudera.com/machine-learning/1.5.5/setup-model-registry/topics/ml-setting-up-model-registry.html
* How to deploy an AI Inference Service in Cloudera AI: https://docs.cloudera.com/machine-learning/1.5.5/setup-cloudera-ai-inference/topics/ml-caii-use-caii.html

### Tutorial

All artifacts are included in this Git repository. You can clone or fork it as needed. https://github.com/pdefusco/CAI_Inf_Service_Articles.git

#### 1. Download the Model from NGC to AI Registry via the Cloudera Model Hub

![alt text](/img/step_1_model_download.png)

![alt text](/img/step_2_model_download.png)

#### 2. Deploy the Model to AI Inference Service

Create a CAI Project and ensure to add the Python 3.11 Cloudera AI Runtime with PBJ Workbench IDE.
