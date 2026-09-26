"""
__init__.py
-----------
This is the ONLY file you touch to "plug in" a new algorithm.

To add algorithm #2 later:
  1. Write a new file, e.g. algorithms/kmeans.py, with a class that
     subclasses BaseAlgorithm and implements get_params/run_steps/build_figure.
  2. Import it below and add one line to ALGORITHM_REGISTRY.

app.py never changes. That's the whole point of the plugin structure.
"""

from algorithms.gradient_descent import GradientDescentLinearRegression

ALGORITHM_REGISTRY = {
    "Gradient Descent (Linear Regression)": GradientDescentLinearRegression,
    # "K-Means Clustering": KMeansClustering,        <- example of future entry
    # "Decision Tree": DecisionTreeVisualizer,       <- example of future entry
}