import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ============================================================
# Monte Carlo vs MISER - Interactive Streamlit Demonstration
# ============================================================

st.set_page_config(
    page_title="Monte Carlo vs MISER",
    page_icon="📊",
    layout="wide",
)

VILLAGE_X = 0.70
VILLAGE_Y = 0.30
VILLAGE_RADIUS = 0.15
SIGMA = 0.055


def density(x, y):
    """Smooth house-density model concentrated near the village."""
    d2 = (x - VILLAGE_X) ** 2 + (y - VILLAGE_Y) ** 2
    return 0.001 + np.exp(-d2 / (2.0 * SIGMA**2))


def in_village(points):
    d2 = (
        (points[:, 0] - VILLAGE_X) ** 2
        + (points[:, 1] - VILLAGE_Y) ** 2
    )
    return d2 <= VILLAGE_RADIUS**2


def reference_integral(grid_size=650):
    """Numerical reference value for the integral over the unit square."""
    axis = np.linspace(0.0, 1.0, grid_size)
    xx, yy = np.meshgrid(axis, axis)
    return float(np.mean(density(xx, yy)))


def run_plain_monte_carlo(n, seed):
    """Uniform sampling over the complete unit square."""
    start = time.perf_counter()
    rng = np.random.default_rng(seed)

    points = rng.random((n, 2))
    values = density(points[:, 0], points[:, 1])
    estimate = float(np.mean(values))

    elapsed = time.perf_counter() - start
    return {
        "points": points,
        "values": values,
        "estimate": estimate,
        "elapsed": elapsed,
    }


def run_miser(n, seed, leaf_size=300):
    """
    MISER-style recursive stratified Monte Carlo.

    At each internal node:
      1) Pilot samples estimate variance in both halves of x and y.
      2) The axis with the lower estimated split cost is selected.
      3) The remaining samples are allocated according to
         subregion area * standard deviation.
      4) The two subregions are integrated recursively.

    Pilot evaluations count toward the total evaluation budget.
    The independent production samples are used for the integral estimate.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)

    all_points = []
    all_kinds = []  # 0 = pilot, 1 = final/production
    split_events = []
    evaluation_count = 0

    def draw(box, count):
        x0, x1 = box[0]
        y0, y1 = box[1]
        pts = np.empty((count, 2), dtype=float)
        pts[:, 0] = rng.uniform(x0, x1, count)
        pts[:, 1] = rng.uniform(y0, y1, count)
        return pts

    def box_area(box):
        return (
            (box[0][1] - box[0][0])
            * (box[1][1] - box[1][0])
        )

    def add_points(points, kind):
        nonlocal evaluation_count
        all_points.append(points)
        all_kinds.append(np.full(len(points), kind, dtype=np.int8))
        evaluation_count += len(points)

    def leaf_estimate(box, count):
        pts = draw(box, count)
        vals = density(pts[:, 0], pts[:, 1])
        add_points(pts, 1)
        return box_area(box) * float(np.mean(vals))

    def recurse(box, count, depth=0):
        nonlocal evaluation_count

        # Pilot samples: m points in each half for x and y tests.
        m = min(16, max(8, count // 80))
        pilot_cost = 4 * m

        # Stop if the region is too small for a reliable split.
        if count <= 2 * leaf_size + pilot_cost or depth >= 12:
            return leaf_estimate(box, count)

        axis_tests = []

        for axis in (0, 1):
            midpoint = (box[axis][0] + box[axis][1]) / 2.0

            left_box = [list(box[0]), list(box[1])]
            right_box = [list(box[0]), list(box[1])]
            left_box[axis][1] = midpoint
            right_box[axis][0] = midpoint

            left_pilot = draw(left_box, m)
            right_pilot = draw(right_box, m)
            add_points(left_pilot, 0)
            add_points(right_pilot, 0)

            left_values = density(left_pilot[:, 0], left_pilot[:, 1])
            right_values = density(right_pilot[:, 0], right_pilot[:, 1])

            var_left = float(np.var(left_values, ddof=1))
            var_right = float(np.var(right_values, ddof=1))

            # Equal-area halves: compare the variance-based split cost.
            split_cost = (var_left ** (1 / 3) + var_right ** (1 / 3)) ** 3

            axis_tests.append({
                "axis": axis,
                "midpoint": midpoint,
                "left_box": left_box,
                "right_box": right_box,
                "var_left": var_left,
                "var_right": var_right,
                "cost": split_cost,
            })

        best = min(axis_tests, key=lambda item: item["cost"])
        remaining = count - pilot_cost

        # Allocate more production samples to the half with larger
        # estimated standard deviation (the two halves have equal area).
        weight_left = np.sqrt(max(best["var_left"], 0.0))
        weight_right = np.sqrt(max(best["var_right"], 0.0))
        weight_sum = weight_left + weight_right

        if weight_sum <= 1e-15:
            n_left = remaining // 2
        else:
            n_left = int(round(remaining * weight_left / weight_sum))

        n_left = max(leaf_size, min(n_left, remaining - leaf_size))
        n_right = remaining - n_left

        split_events.append({
            "axis": best["axis"],
            "midpoint": best["midpoint"],
            "box": [list(box[0]), list(box[1])],
            "depth": depth,
            "after_eval": evaluation_count,
        })

        left_integral = recurse(best["left_box"], n_left, depth + 1)
        right_integral = recurse(best["right_box"], n_right, depth + 1)
        return left_integral + right_integral

    initial_box = [[0.0, 1.0], [0.0, 1.0]]
    estimate = recurse(initial_box, n)

    points = np.vstack(all_points)
    kinds = np.concatenate(all_kinds)
    values = density(points[:, 0], points[:, 1])

    elapsed = time.perf_counter() - start

    # Defensive assertion: every requested function evaluation is counted.
    assert evaluation_count == n, (evaluation_count, n)

    return {
        "points": points,
        "values": values,
        "kinds": kinds,
        "estimate": float(estimate),
        "elapsed": elapsed,
        "splits": split_events,
        "evaluations": evaluation_count,
    }


def make_animated_figure(mc, miser, n, frame_count, frame_duration_ms):
    """Build one synchronized Plotly animation with two side-by-side panels."""
    mc_points = mc["points"]
    miser_points = miser["points"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Plain Monte Carlo — uniform sampling",
            "MISER — adaptive subdivision",
        ),
        horizontal_spacing=0.10,
    )

    # Initial traces. Each algorithm's points are revealed frame by frame.
    fig.add_trace(
        go.Scattergl(
            x=[],
            y=[],
            mode="markers",
            name="MC samples",
            marker=dict(size=4, color="#4C9BE8", opacity=0.55),
            showlegend=False,
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[], y=[],
            mode="lines",
            name="Village boundary",
            line=dict(color="#202020", width=2),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=[],
            y=[],
            mode="markers",
            name="MISER samples",
            marker=dict(size=4, color="#E8754F", opacity=0.58),
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=[], y=[],
            mode="lines",
            name="Village boundary",
            line=dict(color="#202020", width=2),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1, col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=[], y=[],
            mode="lines",
            name="MISER split boundaries",
            line=dict(color="#C56A19", width=1.5),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1, col=2,
    )

    # Village circles are drawn as line traces so both panels use identical axes.
    theta = np.linspace(0, 2 * np.pi, 180)
    circle_x = VILLAGE_X + VILLAGE_RADIUS * np.cos(theta)
    circle_y = VILLAGE_Y + VILLAGE_RADIUS * np.sin(theta)

    fig.data[1].x = circle_x
    fig.data[1].y = circle_y
    fig.data[3].x = circle_x
    fig.data[3].y = circle_y

    # Axis ranges and labels.
    fig.update_xaxes(range=[0, 1], title_text="X", row=1, col=1)
    fig.update_yaxes(range=[0, 1], title_text="Y", row=1, col=1, scaleanchor="x")
    fig.update_xaxes(range=[0, 1], title_text="X", row=1, col=2)
    fig.update_yaxes(range=[0, 1], title_text="Y", row=1, col=2, scaleanchor="x2")

    frames = []
    frame_indices = np.unique(
        np.linspace(1, n, min(frame_count, n), dtype=int)
    )

    split_events = miser["splits"]

    for frame_number, end in enumerate(frame_indices):
        mc_slice = mc_points[:end]
        miser_slice = miser_points[:end]

        # Display MISER split lines only after the corresponding pilot phase.
        line_x, line_y = [], []
        for event in split_events:
            if event["after_eval"] <= end:
                box = event["box"]
                if event["axis"] == 0:
                    x = event["midpoint"]
                    line_x.extend([x, x, None])
                    line_y.extend([box[1][0], box[1][1], None])
                else:
                    y = event["midpoint"]
                    line_x.extend([box[0][0], box[0][1], None])
                    line_y.extend([y, y, None])

        mc_village_pct = (
            float(np.mean(in_village(mc_slice)) * 100)
            if len(mc_slice) else 0.0
        )
        miser_village_pct = (
            float(np.mean(in_village(miser_slice)) * 100)
            if len(miser_slice) else 0.0
        )

        mc_text = (
            f"Samples shown: {end:,}/{n:,}<br>"
            f"Measured runtime: {mc['elapsed']:.5f} s<br>"
            f"Final estimate: {mc['estimate']:.6f}<br>"
            f"Village sample ratio: {mc_village_pct:.1f}%"
        )
        miser_text = (
            f"Evaluations shown: {end:,}/{n:,}<br>"
            f"Measured runtime: {miser['elapsed']:.5f} s<br>"
            f"Final estimate: {miser['estimate']:.6f}<br>"
            f"Village sample ratio: {miser_village_pct:.1f}%"
        )

        frames.append(
            go.Frame(
                name=str(frame_number),
                data=[
                    go.Scattergl(x=mc_slice[:, 0], y=mc_slice[:, 1]),
                    go.Scattergl(x=miser_slice[:, 0], y=miser_slice[:, 1]),
                    go.Scatter(x=line_x, y=line_y),
                ],
                traces=[0, 2, 4],
                layout=go.Layout(
                    annotations=[
                        dict(
                            text=mc_text,
                            x=0.02, y=0.98,
                            xref="x domain", yref="y domain",
                            xanchor="left", yanchor="top",
                            showarrow=False,
                            bgcolor="rgba(255,255,255,0.88)",
                            bordercolor="#B7C7D8",
                            borderwidth=1,
                            font=dict(size=10),
                        ),
                        dict(
                            text=miser_text,
                            x=0.02, y=0.98,
                            xref="x2 domain", yref="y2 domain",
                            xanchor="left", yanchor="top",
                            showarrow=False,
                            bgcolor="rgba(255,255,255,0.88)",
                            bordercolor="#E2B79E",
                            borderwidth=1,
                            font=dict(size=10),
                        ),
                    ]
                ),
            )
        )

    fig.frames = frames

    slider_steps = [
        {
            "args": [
                [str(i)],
                {
                    "frame": {"duration": frame_duration_ms, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": f"{int(frame_indices[i]):,}",
            "method": "animate",
        }
        for i in range(len(frame_indices))
    ]

    fig.update_layout(
        height=650,
        margin=dict(l=35, r=25, t=80, b=95),
        title=dict(
            text="Monte Carlo vs MISER — synchronized sampling process",
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.05,
                "y": -0.16,
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶ Start / Resume",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "Ⅱ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Evaluations shown: "},
                "pad": {"t": 45},
                "steps": slider_steps,
            }
        ],
    )

    # Initial annotations, visible before the user presses Play.
    fig.update_layout(
        annotations=list(fig.layout.annotations) + [
            dict(
                text=(
                    f"Samples shown: 0/{n:,}<br>"
                    f"Measured runtime: {mc['elapsed']:.5f} s<br>"
                    f"Final estimate: {mc['estimate']:.6f}"
                ),
                x=0.02, y=0.98,
                xref="x domain", yref="y domain",
                xanchor="left", yanchor="top",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="#B7C7D8",
                borderwidth=1,
                font=dict(size=10),
            ),
            dict(
                text=(
                    f"Evaluations shown: 0/{n:,}<br>"
                    f"Measured runtime: {miser['elapsed']:.5f} s<br>"
                    f"Final estimate: {miser['estimate']:.6f}"
                ),
                x=0.02, y=0.98,
                xref="x2 domain", yref="y2 domain",
                xanchor="left", yanchor="top",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="#E2B79E",
                borderwidth=1,
                font=dict(size=10),
            ),
        ]
    )

    return fig


def metric_row(label, mc_value, miser_value):
    c1, c2, c3 = st.columns([1.25, 1, 1])
    c1.markdown(f"**{label}**")
    c2.write(mc_value)
    c3.write(miser_value)


# ----------------------------- UI -----------------------------

st.title("📊 Monte Carlo vs MISER")
st.caption(
    "An interactive comparison of uniform Monte Carlo sampling "
    "and variance-guided adaptive subdivision."
)

with st.sidebar:
    st.header("Experiment settings")
    sample_count = st.slider(
        "Function evaluation budget (per algorithm)",
        min_value=2000,
        max_value=20000,
        value=10000,
        step=1000,
    )
    leaf_size = st.slider(
        "MISER leaf sample threshold",
        min_value=100,
        max_value=800,
        value=300,
        step=50,
        help="Smaller values create deeper subdivision; larger values stop recursion earlier.",
    )
    frame_count = st.slider(
        "Animation frames",
        min_value=15,
        max_value=60,
        value=35,
        step=5,
    )
    animation_speed = st.select_slider(
        "Animation speed",
        options=["Slow", "Normal", "Fast"],
        value="Normal",
    )
    speed_ms = {"Slow": 350, "Normal": 180, "Fast": 70}[animation_speed]
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)
    run_button = st.button("▶ Run comparison", type="primary", use_container_width=True)
    reset_button = st.button("↻ Reset results", use_container_width=True)

if reset_button:
    st.session_state.pop("results", None)
    st.rerun()

st.markdown(
    """
    **Model:** a unit-square region containing a circular village centered at
    `(0.70, 0.30)` with radius `0.15`. The density is highest near the village.
    Both algorithms receive the same function-evaluation budget.
    """
)

if run_button:
    progress = st.progress(0, text="Preparing parallel execution...")
    start_parallel = time.perf_counter()

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_mc = pool.submit(
            run_plain_monte_carlo, sample_count, int(seed)
        )
        future_miser = pool.submit(
            run_miser, sample_count, int(seed) + 1, leaf_size
        )

        progress.progress(25, text="Both algorithms are running...")
        mc_result = future_mc.result()
        progress.progress(75, text="Building comparison results...")
        miser_result = future_miser.result()

    parallel_wall_time = time.perf_counter() - start_parallel
    progress.progress(100, text="Experiment completed.")
    progress.empty()

    st.session_state["results"] = {
        "mc": mc_result,
        "miser": miser_result,
        "n": sample_count,
        "leaf_size": leaf_size,
        "frame_count": frame_count,
        "speed_ms": speed_ms,
        "seed": int(seed),
        "parallel_wall_time": parallel_wall_time,
    }

if "results" not in st.session_state:
    st.info(
        "Set the experiment options in the sidebar and click "
        "**Run comparison** to generate the results and animation."
    )

    st.subheader("What you will see")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔵 Plain Monte Carlo")
        st.write("Random points are distributed uniformly across the entire square.")
    with col2:
        st.markdown("### 🟠 MISER")
        st.write(
            "Pilot samples estimate regional variance. MISER recursively "
            "subdivides the region and allocates the remaining samples."
        )
    st.stop()

results = st.session_state["results"]
mc = results["mc"]
miser = results["miser"]
n = results["n"]

# Reference calculation is independent of the randomized algorithms.
truth = reference_integral()
mc_abs_error = abs(mc["estimate"] - truth)
miser_abs_error = abs(miser["estimate"] - truth)
mc_rel_error = mc_abs_error / abs(truth) * 100
miser_rel_error = miser_abs_error / abs(truth) * 100

st.subheader("1. Execution time")
t1, t2, t3 = st.columns(3)
t1.metric("Plain Monte Carlo", f"{mc['elapsed']:.5f} s")
t2.metric("MISER", f"{miser['elapsed']:.5f} s")
t3.metric("Parallel wall time", f"{results['parallel_wall_time']:.5f} s")

st.caption(
    "The two individual times measure each algorithm's computation. "
    "The parallel wall time measures the elapsed time for the concurrent run. "
    "Animation playback time is not included."
)

st.subheader("2. Synchronized sampling animation")
st.write(
    "Use **Start / Resume** and **Pause** inside the chart. "
    "The slider lets you inspect a particular sampling stage. "
    "Both panels advance together."
)

fig = make_animated_figure(
    mc,
    miser,
    n,
    results["frame_count"],
    results["speed_ms"],
)
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

st.subheader("3. Accuracy and sampling comparison")

mc_village_pct = float(np.mean(in_village(mc["points"])) * 100)
miser_village_pct = float(np.mean(in_village(miser["points"])) * 100)

st.markdown("| Metric | Plain Monte Carlo | MISER |")
st.markdown("|:--|--:|--:|")
st.markdown(
    f"| Function evaluations | {len(mc['points']):,} | {len(miser['points']):,} |"
)
st.markdown(
    f"| Estimated integral | {mc['estimate']:.7f} | {miser['estimate']:.7f} |"
)
st.markdown(
    f"| Absolute error | {mc_abs_error:.7f} | {miser_abs_error:.7f} |"
)
st.markdown(
    f"| Relative error | {mc_rel_error:.4f}% | {miser_rel_error:.4f}% |"
)
st.markdown(
    f"| Samples inside village | {mc_village_pct:.2f}% | {miser_village_pct:.2f}% |"
)
st.markdown(
    f"| MISER subdivision count | — | {len(miser['splits']):,} |"
)

st.subheader("4. Interpretation")
if miser_rel_error < mc_rel_error:
    st.success(
        "In this run, MISER produced a smaller relative error under the same "
        "function-evaluation budget."
    )
elif miser_rel_error > mc_rel_error:
    st.info(
        "In this run, Plain Monte Carlo produced a smaller relative error. "
        "Adaptive sampling does not guarantee a better result in every random run."
    )
else:
    st.info("Both methods produced the same relative error in this run.")

st.caption(
    "Important: this is a MISER-style educational implementation, not a "
    "bit-for-bit reproduction of a specific MISER library. Pilot evaluations "
    "are included in the evaluation budget; the independent leaf samples "
    "form the recursive integral estimate. The village sample percentage "
    "counts all evaluated points, including pilot samples."
)
