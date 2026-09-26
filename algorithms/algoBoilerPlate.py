"""
New ML Algorithm .py
-----------------------
[Short description of what this visualizes]
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common.base import BaseAlgorithm
# from sklearn.datasets import load_iris   # only if using a prebuilt dataset


class LogisticRegression(BaseAlgorithm):
    name = "Logistic Regression"
    description = "Fits a decision boundary to classify data by minimizing log loss."

    def __init__(self):
        # Dataset ownership lives here — synthetic OR prebuilt, your choice.
        # e.g. self.X, self.y = make synthetic 2-class blobs with NumPy
        # or:  data = load_iris(); self.X, self.y = data.data[:, :2], (data.target == 0).astype(int)
        pass

    def get_params(self) -> dict:
        # Keys here MUST exactly match run_steps()'s keyword arguments.
        return {
            "learning_rate": (0.001, 1.0, 0.1, 0.001),
            "iterations": (5, 300, 50, 5),
        }

    def run_steps(self, learning_rate, iterations):
        # yield one dict per iteration — whatever state matters for
        # visualizing this algorithm (weights, bias, loss, boundary, etc.)
        for i in range(int(iterations)):
            yield {"iteration": i, "weights": ..., "bias": ..., "loss": ...}
            # update weights/bias here (gradient step)

    def build_figure(self, states: list):
        # turn the collected states into a Plotly figure, ideally animated
        # (decision boundary evolving over the scatter plot is the classic viz)
        fig = go.Figure()
        ...
        return fig