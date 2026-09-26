"""
base.py
-------
This defines the CONTRACT every algorithm in the dashboard must follow.

Why do this? Because app.py (the dashboard) should never need to know
anything specific about "gradient descent" or "k-means" or whatever you
add next. It only knows: "give me your parameters, give me your steps,
give me a figure." As long as every algorithm honors that contract,
app.py never has to change when you add algorithm #5, #10, #50.

This is the "plugin pattern" / "strategy pattern" in OOP terms.
"""

from abc import ABC, abstractmethod
from typing import Generator


class BaseAlgorithm(ABC):
    # Human-readable name shown in the dashboard dropdown.
    name: str = "Unnamed Algorithm"

    # Short description shown under the dropdown.
    description: str = ""

    @abstractmethod
    def get_params(self) -> dict:
        """
        Declare the tunable inputs this algorithm needs, so the dashboard
        can auto-generate sliders for them.

        Return format:
            {
                "param_name": (min_value, max_value, default_value, step),
                ...
            }

        Example:
            {"learning_rate": (0.001, 1.0, 0.1, 0.001)}
        """
        raise NotImplementedError

    @abstractmethod
    def run_steps(self, **params) -> Generator[dict, None, None]:
        """
        Run the algorithm and YIELD one dictionary per step/iteration,
        describing its internal state at that moment.

        This is a generator (uses `yield`, not `return`) so that:
          1. We can visualize progress step-by-step, not just the final answer.
          2. Long-running algorithms don't block everything until fully done.

        The keys in each yielded dict are entirely up to the algorithm —
        app.py doesn't inspect them. Only build_figure() (below) needs to
        know the shape of this data, because the same class writes both.
        """
        raise NotImplementedError

    @abstractmethod
    def build_figure(self, states: list):
        """
        Given the full list of per-step states collected from run_steps(),
        build and return a Plotly Figure (go.Figure) — ideally an animated
        one with a play/pause control — that visualizes how the algorithm
        progressed.

        This is where "the math" becomes "the picture."
        """
        raise NotImplementedError