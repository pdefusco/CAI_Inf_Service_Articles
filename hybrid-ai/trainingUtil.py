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

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SimpleTrainingPipeline:
    """A compact pipeline that builds data, trains a small classifier,
    evaluates it, and prepares example inputs."""

    def __init__(self, num_samples=20, lr=0.01, epochs=200):
        self.num_samples = num_samples
        self.lr = lr
        self.epochs = epochs

        # Prepare data immediately when pipeline is created
        self.X, self.y = self._generate_data()
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    # ----------------------------------------
    # Internal helpers
    # ----------------------------------------
    def _generate_data(self):
        """Generate a tiny synthetic dataset with two 2D Gaussian clusters."""
        class0 = torch.randn(self.num_samples, 2) * 0.3 + torch.tensor([-1.0, -1.0])
        class1 = torch.randn(self.num_samples, 2) * 0.3 + torch.tensor([1.0, 1.0])

        X = torch.cat([class0, class1], dim=0)
        y = torch.cat([
            torch.zeros(self.num_samples),
            torch.ones(self.num_samples)
        ]).long()

        return X, y

    def _build_model(self):
        """Define a tiny MLP classifier."""
        class SimpleClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, 16),
                    nn.ReLU(),
                    nn.Linear(16, 2)
                )

            def forward(self, x):
                return self.net(x)

        return SimpleClassifier()

    def _train(self):
        """Training loop with periodic reporting."""
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            logits = self.model(self.X)
            loss = self.criterion(logits, self.y)

            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 40 == 0:
                pred = torch.argmax(logits, dim=1)
                acc = (pred == self.y).float().mean().item()
                print(f"Epoch {epoch+1:3d} | Loss = {loss.item():.4f} | Acc = {acc*100:.1f}%")

    def _test(self):
        """Evaluate model on a few unseen points."""
        test_points = torch.tensor([
            [-1.2, -0.8],
            [1.1,  0.9],
            [0.0,  0.0]
        ])

        with torch.no_grad():
            out = self.model(test_points)
            preds = torch.argmax(out, dim=1)
            print("\nTest predictions:", preds.tolist())

    def _prepare_input_example(self):
        """Prepare a small NumPy + Torch example batch for downstream logging."""
        input_example = np.array([
            [-0.66, -1.77],
            [-0.74, -1.42],
            [-1.50, -0.92]
        ], dtype=np.float32)

        input_tensor = torch.from_numpy(input_example).float()

        print(f"Input example shape: {input_example.shape}")
        print(f"Input example (first sample): {input_example[0]}")

        return input_example, input_tensor

    def run(self):
        """Run the complete training → testing → example-prep workflow.

        Returns:
            model (nn.Module): The trained PyTorch model.
            input_example (np.ndarray): Example NumPy input batch.
            input_tensor (torch.Tensor): Tensor version of input_example.
        """
        self._train()
        self._test()
        input_example, input_tensor = self._prepare_input_example()
        return self.X, self.model, input_example, input_tensor
