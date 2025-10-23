# Deploy a PyTorch Model with Cloudera AI Inference Service

This is WIP.

## Objective

In this tutorial you will learn how to deploy Deepseek R1 Distill Llama 8B to the Cloudera AI Inference Service programmatically using Python and an LLMOps Util.

First, you will download the model to the Cloudera AI Registry; Then, you will create an AI Inference Service Endpoint to serve predictions in real time from within your Data Center.

The LLMOps util and the overall tutorial is particularly tailored for CAI users who want to complete the end to end lifecycle - from LLM download to Endpoint - entirely in Python. If you'd like to use a different model you can apply the same steps for any language models available in the NGC Catalog.  

### Motivation

With Cloudera AI, enterprises can download open source GenAI models and securely host them in their Public or Private Cloud, in order to implement LLM-powered applications while preventing proprietary information from being shared with LLM Service Providers such as OpenAI.

### Cloudera AI & LLM's

Cloudera AI (CAI) is a platform that enables organizations to build, train, and deploy machine learning and artificial intelligence models at scale. One of its key features is the Cloudera AI Inference Service, which allows users to easily deploy large language models (LLMs) for real-time or batch inference. With Cloudera AI, data scientists and engineers can manage and serve LLMs like Llama, Mistral, or open-source GPT models using containerized environments and scalable infrastructure. This service supports secure, low-latency model serving, making it easier to integrate AI into enterprise applications.

### Hybrid Enterprise AI with CAI

Cloudera AI (CAI) is a core component of Cloudera’s hybrid cloud data platform, which is designed to operate seamlessly across both private and public cloud environments. This hybrid architecture allows organizations to deploy AI models securely wherever their data resides—on-premises for sensitive workloads or in the public cloud for greater scalability and flexibility. With Cloudera AI, enterprises can maintain governance, compliance, and control over their machine learning pipelines while taking advantage of cloud-native capabilities. This ensures that large language models and other AI applications can be deployed and managed securely across diverse IT environments without compromising performance or data privacy.

### CAI Integration with NGC

NVIDIA NGC (NVIDIA GPU Cloud) is a catalog of GPU-optimized AI, machine learning, and HPC software, including pre-trained models, model training scripts, and containers, designed to accelerate AI development and deployment. The integration of NVIDIA NGC with Cloudera's AI Model Hub and AI Registry enables seamless access to NGC’s curated models and resources within Cloudera’s unified AI platform. This integration streamlines the workflow for data scientists and ML engineers, allowing them to discover, import, fine-tune, and manage NGC models directly within Cloudera’s secure and governed enterprise environment, accelerating the path from experimentation to production.

## Requirements

This example was built with Cloudera On Cloud Public Cloud Runtime 7.3.1, CAI Workbench 2.0.50, Inference Service 1.4.0 and AI Registry 1.7.0.

The same example will also work in Private Cloud aside from having to use root certs in the httpx client class. For an example of this, check the way the client is instantiated in [llama-hf/model-deployment.ipynb](https://github.com/pdefusco/CAI_Inf_Service_Articles/blob/main/llama-hf/model-deployment.ipynb).  

You can reproduce this tutorial in your CAI environment with the following:

* A local installation of the CDP CLI.
* A CAI Environment in Private or Public Cloud.
* An AI Registry deployment.
* An AI Inference Service deployment.
* The following ML Runtime should be available in your CAI Runtime Catalog and Project: ```pauldefusco/vscode4_cuda11_cml_runtime```
* You should have your CDP Workload User's Access Keys handy as you will need them to deploy the endpoint.

If you have an airgapped environment with Cloudera On Prem, you can still follow the steps shown in the the model deployment JupyterLab notebook as long as you have already completed the extra steps to download the model with the python script shown in the documentation: https://docs.cloudera.com/machine-learning/1.5.5/importing-model-airgapped/topics/ml-models-in-air-gapped-environment.html

## Useful Documentation Links

* How to deploy a Workbench in Cloudera AI: https://docs.cloudera.com/machine-learning/1.5.5/workspaces-privatecloud/topics/ml-pvc-provision-ml-workspace.html
* How to deploy an AI Registry in Cloudera AI: https://docs.cloudera.com/machine-learning/1.5.5/setup-model-registry/topics/ml-setting-up-model-registry.html
* How to deploy an AI Inference Service in Cloudera AI: https://docs.cloudera.com/machine-learning/1.5.5/setup-cloudera-ai-inference/topics/ml-caii-use-caii.html
* How to set up an NGC account:
* How to set up the CDP CLI: https://docs.cloudera.com/cdp-public-cloud/cloud/cli/topics/mc-cdp-cli.html

### Tutorial


#### 1. Import VSCode Runtime in Runtime Catalog

#### 2. Download Deepseek via Cloudera AI Model Hub

#### 3. Clone the Git Repository as a CAI Project

![alt text](../img/project-wizard-2.png)

#### 4. Launch Session and Install Requirements

#### 5. Run the Llmops App

## Summary & Next Steps

In this tutorial,
