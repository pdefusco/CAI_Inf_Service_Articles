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

import os
import json
import httpx
import sys
import subprocess
import requests

# Use this if /tmp/jwt is supported by the registry and there's no
# misconfiguration of the DL Knox gateway
def get_jwt_token():
    try:
        with open("/tmp/jwt", "r") as jwt_file:
            jwt_data = json.load(jwt_file)
            print(jwt_data["access_token"])
            return jwt_data["access_token"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error: {e}")
        exit(1)

class ModelRegistryClient:
    def __init__(self, base_url: str, bearer_token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def list_models(self):
        """
        Fetch the list of models from the Model Registry microservice.
        """
        if "Content-Type" in self.headers:
            del self.headers["Content-Type"]
        url = f"{self.base_url}/api/v2/models"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def create_model(
        self,
        name: str,
        repo_id: str,
        hf_token: str,
        visibility: str = "public",
        model_repo_type: str = "HF"
    ):
        if "Content-Type" not in self.headers:
          self.headers["Content-Type"] = "application/json"
        url = f"{self.base_url}/api/v2/models"
        payload = {
            "name": name,
            "createModelVersionRequestPayload": {
                "metadata": {
                    "model_repo_type": model_repo_type
                },
                "downloadModelRepoRequest": {
                    "source": model_repo_type,
                    "repo_id": repo_id,
                    "hf_token": hf_token
                }
            },
            "domain": self.base_url,
            "visibility": visibility
        }

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()


def create_client():
    # Replace get_jwt_token() with get_ums_jwt_token() if needed
    TOKEN = os.environ["CDP_TOKEN"]
    print(TOKEN)
    BASE_URL = "https://modelregistry.[...].cloudera.site"
    client = ModelRegistryClient(base_url=BASE_URL, bearer_token=TOKEN)

#    list all the models
    try:
        models = client.list_models()
        for model in models.get("models", []):
            print(f"{model['name']} (ID: {model['id']})")
    except requests.HTTPError as e:
        print(f"Failed to list models: {e}")

#     create a new model
    new_model = client.create_model(
        name="",
        repo_id="",
        hf_token=""
    )
    print("✅ Model Created:\n", new_model)


def main():
    client = create_client()

if __name__ == "__main__":
  main()
