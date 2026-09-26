"""
gradient_descent.py
--------------------
Our FIRST plugin. It implements linear regression trained with plain
batch gradient descent, and visualizes two things side by side:

  Left panel:  the loss surface (a bowl-shaped contour plot in
               (slope, intercept) space) with a red path showing how
               gradient descent walks downhill toward the minimum.

  Right panel: the actual data points and the fitted line, updating
               at each step, so you can SEE the line rotate/shift
               into place as the loss decreases.

This file only needs to satisfy the 3 methods in BaseAlgorithm.
Everything else (sliders, run button, animation controls) is handled
generically by app.py.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common.base import BaseAlgorithm


class GradientDescentLinearRegression(BaseAlgorithm):
    name = "Gradient Descent (Linear Regression)"
    description = (
        "Fits a line y = w*x + b to noisy data by repeatedly stepping "
        "downhill on the mean-squared-error loss surface."
    )

    def __init__(self, n_points: int = 40, seed: int = 42):
        # This algorithm generates its own synthetic dataset inline.
        # Each algorithm file owns its dataset logic — whether that means
        # generating synthetic data (like here) or loading a prebuilt
        # dataset (e.g. sklearn.datasets.load_iris in a future algorithm).
        rng = np.random.default_rng(seed)
        self.x = rng.uniform(0, 10, n_points)
        true_w, true_b = 3.0, 5.0
        noise = rng.normal(0, 3, n_points)
        self.y = true_w * self.x + true_b + noise

    # ------------------------------------------------------------------
    # 1. Declare tunable parameters -> dashboard auto-builds sliders
    # ------------------------------------------------------------------
    def get_params(self) -> dict:
        return {
            "learning_rate": (0.001, 0.05, 0.01, 0.001),
            "iterations": (5, 200, 50, 5),
            "start_w": (-10.0, 10.0, -8.0, 0.5),
            "start_b": (-10.0, 10.0, -8.0, 0.5),
        }

    # ------------------------------------------------------------------
    # 2. The actual algorithm, yielding state at every step
    # ------------------------------------------------------------------
    def run_steps(self, learning_rate, iterations, start_w, start_b):
        w, b = start_w, start_b
        n = len(self.x)

        for i in range(int(iterations)):
            # Model prediction with current parameters
            y_pred = w * self.x + b

            # Mean squared error loss
            error = y_pred - self.y
            loss = np.mean(error ** 2)

            # Gradients of MSE loss w.r.t. w and b (calculus, precomputed)
            grad_w = (2 / n) * np.sum(error * self.x)
            grad_b = (2 / n) * np.sum(error)

            # Yield the state BEFORE updating, so step 0 shows the
            # starting point, not the first update already applied.
            yield {
                "iteration": i,
                "w": w,
                "b": b,
                "loss": loss,
            }

            # Gradient descent update rule: step opposite the gradient
            w -= learning_rate * grad_w
            b -= learning_rate * grad_b

    # ------------------------------------------------------------------
    # 3. Turn the collected states into an animated Plotly figure
    # ------------------------------------------------------------------
    def build_figure(self, states: list):
        ws = [s["w"] for s in states]
        bs = [s["b"] for s in states]
        losses = [s["loss"] for s in states]

        # --- Build the loss surface as a grid around the path taken ---
        w_range = np.linspace(min(ws) - 2, max(ws) + 2, 60)
        b_range = np.linspace(min(bs) - 2, max(bs) + 2, 60)
        W, B = np.meshgrid(w_range, b_range)

        # Vectorized loss computation over the whole grid at once
        n = len(self.x)
        Loss = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                pred = W[i, j] * self.x + B[i, j]
                Loss[i, j] = np.mean((pred - self.y) ** 2)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Loss surface + descent path", "Data + fitted line"),
        )

        # Static background: the contour plot (doesn't change per frame)
        fig.add_trace(
            go.Contour(x=w_range, y=b_range, z=Loss, showscale=False,
                       colorscale="Blues", opacity=0.8),
            row=1, col=1,
        )
        # Static background: the raw data points
        fig.add_trace(
            go.Scatter(x=self.x, y=self.y, mode="markers",
                       marker=dict(color="gray"), name="data"),
            row=1, col=2,
        )

        # Dynamic traces that will be updated frame-by-frame:
        # trace index 2 = path so far on the contour plot
        fig.add_trace(
            go.Scatter(x=[ws[0]], y=[bs[0]], mode="lines+markers",
                       line=dict(color="red", width=2),
                       marker=dict(size=6, color="red"), name="path"),
            row=1, col=1,
        )
        # trace index 3 = current fitted line on the data plot
        line_x = np.array([self.x.min(), self.x.max()])
        fig.add_trace(
            go.Scatter(x=line_x, y=ws[0] * line_x + bs[0], mode="lines",
                       line=dict(color="red", width=3), name="fitted line"),
            row=1, col=2,
        )

        # --- Build one animation frame per iteration ---
        frames = []
        for i in range(len(states)):
            frames.append(go.Frame(
                name=str(i),
                data=[
                    go.Contour(x=w_range, y=b_range, z=Loss, showscale=False,
                               colorscale="Blues", opacity=0.8),
                    go.Scatter(x=self.x, y=self.y, mode="markers",
                               marker=dict(color="gray")),
                    go.Scatter(x=ws[:i + 1], y=bs[:i + 1], mode="lines+markers",
                               line=dict(color="red", width=2),
                               marker=dict(size=6, color="red")),
                    go.Scatter(x=line_x, y=ws[i] * line_x + bs[i],
                               mode="lines", line=dict(color="red", width=3)),
                ],
            ))
        fig.frames = frames

        # --- Play/pause buttons + step slider ---
        fig.update_layout(
            height=500,
            showlegend=False,
            title=f"Final loss: {losses[-1]:.3f}",
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {"label": "▶ Play", "method": "animate",
                     "args": [None, {"frame": {"duration": 150, "redraw": True},
                                      "fromcurrent": True}]},
                    {"label": "⏸ Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
                ],
            }],
            sliders=[{
                "steps": [
                    {"args": [[str(i)], {"frame": {"duration": 0, "redraw": True},
                                          "mode": "immediate"}],
                     "label": str(i), "method": "animate"}
                    for i in range(len(states))
                ],
                "currentvalue": {"prefix": "Iteration: "},
            }],
        )
        fig.update_xaxes(title_text="w (slope)", row=1, col=1)
        fig.update_yaxes(title_text="b (intercept)", row=1, col=1)
        fig.update_xaxes(title_text="x", row=1, col=2)
        fig.update_yaxes(title_text="y", row=1, col=2)

        return fig