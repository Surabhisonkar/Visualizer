"""
app.py
------
The dashboard shell. Notice this file contains ZERO knowledge of gradient
descent, MSE, or linear regression. It only knows about the BaseAlgorithm
contract (get_params, run_steps, build_figure). That's what makes it
generic — you'll add many algorithms without ever editing this file.

Run with:  streamlit run app.py
"""

import streamlit as st
from algorithms import ALGORITHM_REGISTRY

st.set_page_config(page_title="ML Algorithm Visualizer", layout="wide")
st.title("🧠 ML Algorithm Visualizer")
st.caption("Pick an algorithm, tune it, and watch it learn step by step.")

# ----------------------------------------------------------------------
# 1. Algorithm selection (dropdown)
# ----------------------------------------------------------------------
algo_name = st.sidebar.selectbox("Choose an algorithm", list(ALGORITHM_REGISTRY.keys()))
algo_class = ALGORITHM_REGISTRY[algo_name]
algo = algo_class()  # instantiate the chosen algorithm

st.sidebar.markdown(f"**About:** {algo.description}")

# ----------------------------------------------------------------------
# 2. Auto-generate sliders from whatever params THIS algorithm declares
# ----------------------------------------------------------------------
st.sidebar.markdown("### Parameters")
params = {}
for param_name, (lo, hi, default, step) in algo.get_params().items():
    label = param_name.replace("_", " ").title()
    params[param_name] = st.sidebar.slider(
        label, min_value=lo, max_value=hi, value=default, step=step
    )

# ----------------------------------------------------------------------
# 3. Run + visualize
# ----------------------------------------------------------------------
run_clicked = st.sidebar.button("▶ Run Algorithm", type="primary")

if run_clicked:
    with st.spinner("Running algorithm..."):
        # Collect every step's state by exhausting the generator.
        states = list(algo.run_steps(**params))
        # Hand the full history to the algorithm's own figure builder.
        fig = algo.build_figure(states)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("See raw step-by-step data"):
        st.dataframe(states)
else:
    st.info("Set your parameters in the sidebar, then click **Run Algorithm**.")