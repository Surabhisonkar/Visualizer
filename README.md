# ML Algorithm Visualizer

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens a browser tab with the dashboard.

## Project structure

```
visualizer/
├── app.py                       # Dashboard shell — generic, algorithm-agnostic
├── requirements.txt
└── algorithms/
    ├── __init__.py               # Registry: add new algorithms here
    ├── base.py                   # The contract every algorithm follows
    └── gradient_descent.py       # First algorithm: linear regression via gradient descent
```

## How to add your next algorithm

1. Create `algorithms/your_algorithm.py`.
2. Subclass `BaseAlgorithm` (from `algorithms/base.py`) and implement:
   - `get_params()` — declares sliders
   - `run_steps(**params)` — a generator yielding one state dict per step
   - `build_figure(states)` — returns a Plotly figure (ideally animated)
3. Import your class in `algorithms/__init__.py` and add it to `ALGORITHM_REGISTRY`.
4. Done — `app.py` picks it up automatically, no other changes needed.