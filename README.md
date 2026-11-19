# Monte Carlo vs MISER — Streamlit App

Interactive web demonstration comparing uniform Monte Carlo sampling with a
MISER-style adaptive recursive Monte Carlo integrator.

## Project files

- `app.py` — Streamlit application, algorithms, metrics, and synchronized Plotly animation
- `requirements.txt` — Python dependencies

## Run locally

Python 3.10–3.12 is recommended.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository (for example, `monte-carlo-miser`).
2. Upload `app.py` and `requirements.txt` to the repository root.
3. Sign in at https://share.streamlit.io/ using GitHub.
4. Select **Create app**.
5. Choose the repository, branch (usually `main`), and main file path `app.py`.
6. Click **Deploy**.
7. After deployment, copy the generated `*.streamlit.app` URL for your presentation.

The repository must be accessible to the Streamlit account. For deployment, you
need admin access to the GitHub repository.

## How to use

1. Choose the function-evaluation budget, MISER leaf threshold, animation frames,
   animation speed, and random seed in the sidebar.
2. Click **Run comparison**.
3. Wait for both algorithms to finish their parallel computation.
4. Use **Start / Resume**, **Pause**, and the slider in the Plotly chart to inspect
   the synchronized sampling process.
5. Compare individual runtime, integral estimate, absolute/relative error, and
   village sample ratio.

## Important interpretation notes

- Both algorithms receive the same total number of function evaluations.
- The Plain Monte Carlo estimate is the sample mean over uniform points in the
  unit square (whose area is 1).
- MISER uses pilot samples to estimate variance in candidate subregions, chooses
  a split direction, allocates remaining samples, and recursively integrates
  the subregions.
- Pilot evaluations count toward MISER's evaluation budget. The production
  samples at terminal regions are used for the recursive integral estimate.
- The reported village percentage counts all evaluated MISER points, including
  pilot points. It is a sampling diagnostic, not the fraction used directly in
  the final integral estimate.
- The reference integral is a numerical grid approximation, not an analytic
  exact value.
- Runtime depends on the hosting CPU and concurrent workload. MISER is not
  guaranteed to be faster or more accurate in every random run.
- Plotly animation playback occurs after the computation. The displayed
  algorithm runtimes are measured computation times, not animation duration.
