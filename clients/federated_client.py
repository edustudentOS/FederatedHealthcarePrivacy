

from pathlib import Path

import flwr as fl
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


class HealthcareClient(fl.client.NumPyClient):
    def __init__(self, hospital_id):
        self.hospital_id = hospital_id

        data_path = (
            PROJECT_DIR
            / "data"
            / "clients"
            / f"hospital_{hospital_id}.csv"
        )

        df = pd.read_csv(data_path)
        self.X = df.drop(columns=["target"]).to_numpy(dtype=np.float64)
        self.y = df["target"].to_numpy(dtype=np.float64)

        # Add bias column for the intercept.
        self.X = np.column_stack(
            [self.X, np.ones(len(self.X), dtype=np.float64)]
        )

        self.num_samples = len(self.y)
        self.weights = np.zeros(self.X.shape[1], dtype=np.float64)

    def get_parameters(self, config):
        return [self.weights.copy()]

    def set_parameters(self, parameters):
        self.weights = parameters[0].astype(np.float64).copy()

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Local gradient-descent training.
        learning_rate = 0.1
        epochs = 10

        for _ in range(epochs):
            probabilities = sigmoid(self.X @ self.weights)
            gradient = (self.X.T @ (probabilities - self.y)) / self.num_samples
            self.weights -= learning_rate * gradient

        print(
            f"Hospital {self.hospital_id}: "
            f"completed local training on {self.num_samples} rows"
        )

        return self.get_parameters(config), self.num_samples, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        probabilities = sigmoid(self.X @ self.weights)
        eps = 1e-12

        loss = -np.mean(
            self.y * np.log(probabilities + eps)
            + (1 - self.y) * np.log(1 - probabilities + eps)
        )

        predictions = (probabilities >= 0.5).astype(int)
        accuracy = float(np.mean(predictions == self.y))

        return float(loss), self.num_samples, {"accuracy": accuracy}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hospital",
        type=int,
        required=True,
        choices=[1, 2, 3],
    )
    args = parser.parse_args()

    client = HealthcareClient(args.hospital)

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
