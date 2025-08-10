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

# Import inference-related libraries
import httpx
import json
import time
import os
import subprocess
from urllib.parse import urlparse, urlunparse
from typing import Optional, Dict, Any, List
import getpass
from llmopsUtils import ModelRegistryClient, Llmops

llmopsClient = Llmops()

# Configure CDP control plane credentials
access_key_id = getpass.getpass("Enter your CDP access key ID: ")
private_key = getpass.getpass("Enter your CDP private key: ")
llmopsClient.configure_cdp("cdp_access_key_id", access_key_id)
llmopsClient.configure_cdp("cdp_private_key", private_key)

ENVIRONMENT_NAME = os.environ["ENVIRONMENT_NAME"] # Enter CDP env name here e.g. "pdf-jul-25-cdp-env"
REGISTERED_MODEL_NAME = os.environ["ENVIRONMENT_NAME"] # Enter model name as you'd like it to appear in AI Registry e.g. "mixtral-8x7b-instruct"
HF_REPO_ID = os.environ["ENVIRONMENT_NAME"] # Enter Repo ID for model as it appears in HF Catalog e.g. "mistralai/Mixtral-8x7B-Instruct-v0.1"
ENDPOINT_NAME = os.environ["ENVIRONMENT_NAME"] # Enter endpoint name as you'd like it to appear in AIIS e.g. "mixtral-endpoint"
HF_TOKEN = os.environ["HF_TOKEN"] # Create Project Env Var with your HF Catalog Token, or set it directly here

CAII_DOMAIN = llmopsClient.get_caii_domain(ENVIRONMENT_NAME)
CDP_TOKEN = llmopsClient.get_ums_jwt_token()
print("CAI DOMAIN: ", CAII_DOMAIN)
print("CDP TOKEN: ", CDP_TOKEN)

registryClient = ModelRegistryClient(base_url=REGISTRY_ENDPOINT, bearer_token=TOKEN)

REGISTRY_ENDPOINT = registryClient.get_registry_endpoint(ENVIRONMENT_NAME)
print("REGISTRY ENDPOINT: ", REGISTRY_ENDPOINT)

new_model = registryClient.create_model(
    name=REGISTERED_MODEL_NAME,
    repo_id=HF_REPO_ID,
    hf_token=HF_TOKEN
)

model_details = llmopsClient.get_model_details(REGISTRY_ENDPOINT, REGISTERED_MODEL_NAME, CDP_TOKEN)
print(model_details)

MODEL_VERSION = llmopsClient.get_most_recent_model_version(REGISTRY_ENDPOINT, model_details['id'], CDP_TOKEN)

MODEL_ID = model_details['id']
print(MODEL_ID)
print(MODEL_VERSION)

llmopsClient.deploy_model_to_caii(CAII_DOMAIN, CDP_TOKEN, MODEL_ID, MODEL_VERSION, ENDPOINT_NAME)

# Must return True before we go on to the next step
ready = llmopsClient.endpoint_is_ready(CAII_DOMAIN, CDP_TOKEN, ENDPOINT_NAME)
print(ready)

BASE_URL = llmopsClient.get_endpoint_base_url(CAII_DOMAIN, CDP_TOKEN, ENDPOINT_NAME)
print(BASE_URL)
