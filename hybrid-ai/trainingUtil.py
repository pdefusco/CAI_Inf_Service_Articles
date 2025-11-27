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

