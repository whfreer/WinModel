import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="4th Down Decision Heatmaps", layout="wide")


@st.cache_data
def load_precomputed_data():
    """Load pre-computed heatmap data with caching"""
    try:
        # Check if data files exist
        if not os.path.exists('heatmap_data.pkl'):
            st.error("Pre-computed data not found. Please run the data pre-calculation script first.")
            st.info("Run: `python precalculate_data.py` to generate the required data files.")
            return None, None

        # Load the main data
        with open('heatmap_data.pkl', 'rb') as f:
            all_scenario_data = pickle.load(f)

        # Load metadata if available
        metadata = None
        if os.path.exists('scenario_metadata.json'):
            with open('scenario_metadata.json', 'r') as f:
                metadata = json.load(f)

        logger.info(f"Loaded pre-computed data for {len(all_scenario_data)} scenarios")
        return all_scenario_data, metadata

    except Exception as e:
        logger.error(f"Error loading pre-computed data: {e}")
        st.error(f"Error loading data: {str(e)}")
        return None, None


def create_heatmap_from_data(data, title=""):
    """Create heatmap figure from pre-computed data"""
    try:
        # Extract pre-computed data
        heat = data['go_for_it_score_grid']
        cell_raw_probs = data['cell_raw_probs']
        cell_threshold_info = data['cell_threshold_info']
        yards_to_go = data['yards_to_go']
        yards_to_goals = data['yards_to_goals']
        max_shift = data['max_shift']
        params = data['parameters']

        # Process the grid for display
        x_values = np.arange(-max_shift, len(yards_to_goals))
        valid_mask = (x_values >= 0) & (x_values <= 89)
        x_values_filtered = x_values[valid_mask]
        heat_filtered = heat[:, valid_mask]
        nrows, ncols = heat_filtered.shape

        # Create hover text
        hover_text = np.full(heat_filtered.shape, "", dtype=object)
        for i, distance in enumerate(yards_to_go):
            for j in range(ncols):
                yards_between = int(x_values_filtered[j])

                # Try to find matching data
                possible_keys = [
                    (int(distance), yards_between),
                    (int(distance), yards_between - 1),
                    (int(distance), yards_between + 1),
                ]

                found_data = False
                for key in possible_keys:
                    if key in cell_raw_probs and not np.isnan(heat_filtered[i, j]):
                        probs = cell_raw_probs[key]
                        threshold_info = cell_threshold_info.get(key, [])

                        hover_text[i, j] = (
                                f"Distance: {distance} yards<br>"
                                f"Yards Between: {key[1]}<br>"
                                f"Field Position: {key[1] + distance} yard line<br><br>"
                                f"Probabilities:<br>"
                                f"Punt: {probs['Punt']:.2f}<br>"
                                f"Field Goal: {probs['Field Goal']:.2f}<br>"
                                f"Go for It: {probs['Go for It']:.2f}<br><br>"
                                f"Threshold Recommendations:<br>" +
                                "<br>".join(threshold_info)
                        )
                        found_data = True
                        break

                if not found_data:
                    hover_text[
                        i, j] = f"Distance: {distance} yards<br>Yards Between: {yards_between}<br>No data available"

        # Enhanced discrete colors for 5 levels
        colorscale = [
            [0.0, "#d32f2f"],  # Red - Punt
            [0.199, "#d32f2f"],
            [0.20, "#ffa726"],  # Orange/Yellow - Field Goal
            [0.399, "#ffa726"],
            [0.40, "#a5d6a7"],  # Very Light Green - 50-50 Go
            [0.599, "#a5d6a7"],
            [0.60, "#66bb6a"],  # Medium Green - Recommend Go
            [0.799, "#66bb6a"],
            [0.80, "#2e7d32"],  # Dark Green - Must Go
            [1.0, "#2e7d32"]
        ]

        # Split into two halves
        midpoint = ncols // 2
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                "Recommended 4th Down Decision (Opponent Territory)",
                "Recommended 4th Down Decision (Own Territory)"
            )
        )

        x_indices = list(range(ncols))

        # Opponent territory (first half)
        fig.add_trace(go.Heatmap(
            z=heat_filtered[:, :midpoint],
            x=x_indices[:midpoint],
            y=yards_to_go,
            text=hover_text[:, :midpoint],
            hovertemplate='%{text}<extra></extra>',
            colorscale=colorscale,
            zmin=0, zmax=4,
            showscale=True,
            colorbar=dict(
                tickvals=[0, 1, 2, 3, 4],
                ticktext=["Punt", "Field Goal", "Conservative Go", "Recommend Go", "Aggressive Go"],
                title="Decision",
                len=0.8
            )
        ), row=1, col=1)

        # Own territory (second half)
        fig.add_trace(go.Heatmap(
            z=heat_filtered[:, midpoint:],
            x=x_indices[:ncols - midpoint],
            y=yards_to_go,
            text=hover_text[:, midpoint:],
            hovertemplate='%{text}<extra></extra>',
            colorscale=colorscale,
            zmin=0, zmax=4,
            showscale=False
        ), row=2, col=1)

        # Update axes
        fig.update_yaxes(title="Yards to Go", autorange="reversed", row=1, col=1)
        fig.update_yaxes(title="Yards to Go", autorange="reversed", row=2, col=1)

        # Custom x-axis labels
        opponent_ticks = list(range(midpoint))
        opponent_labels = [str(x_values_filtered[i]) for i in range(midpoint)]
        fig.update_xaxes(
            title="Yards Between First Down Marker and Goal Line",
            tickvals=opponent_ticks,
            ticktext=opponent_labels,
            row=1, col=1
        )

        own_ticks = list(range(ncols - midpoint))
        own_labels = [str(x_values_filtered[i + midpoint]) for i in range(ncols - midpoint)]
        fig.update_xaxes(
            title="Yards Between First Down Marker and Goal Line",
            tickvals=own_ticks,
            ticktext=own_labels,
            row=2, col=1
        )

        fig.update_layout(
            title=f"{title}  |  Score Diff: {params['score_diff']}  |  Time: {params['time']}s",
            margin=dict(t=100, b=50, l=60, r=140),
            hoverlabel=dict(align="left", bgcolor="white", font_size=12),
            autosize=True,
            height=800
        )

        # Add gridlines
        line_color = "rgba(0,0,0,0.3)"
        line_width = 0.5

        # Gridlines for opponent territory
        for col in range(midpoint + 1):
            fig.add_shape(
                type="line",
                x0=col - 0.5, x1=col - 0.5,
                y0=0.5, y1=len(yards_to_go) + 0.5,
                line=dict(color=line_color, width=line_width),
                row=1, col=1
            )

        for r in range(nrows + 1):
            fig.add_shape(
                type="line",
                x0=-0.5, x1=midpoint - 0.5,
                y0=r + 0.5, y1=r + 0.5,
                line=dict(color=line_color, width=line_width),
                row=1, col=1
            )

        # Gridlines for own territory
        own_ncols = ncols - midpoint
        for col in range(own_ncols + 1):
            fig.add_shape(
                type="line",
                x0=col - 0.5, x1=col - 0.5,
                y0=0.5, y1=len(yards_to_go) + 0.5,
                line=dict(color=line_color, width=line_width),
                row=2, col=1
            )

        for r in range(nrows + 1):
            fig.add_shape(
                type="line",
                x0=-0.5, x1=own_ncols - 0.5,
                y0=r + 0.5, y1=r + 0.5,
                line=dict(color=line_color, width=line_width),
                row=2, col=1
            )

        return fig

    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
        st.error(f"Error creating heatmap: {str(e)}")
        return None


# Main app
def main():
    st.title("Multiple 4th Down Decision Heatmaps")

    # Load pre-computed data
    with st.spinner("Loading pre-computed data..."):
        all_scenario_data, metadata = load_precomputed_data()

    if all_scenario_data is None:
        st.stop()

    # Show data info
    if metadata:
        st.sidebar.markdown(f"## Data Info")
        st.sidebar.markdown(f"**Scenarios loaded:** {metadata['total_scenarios']}")
        if 'created_at' in metadata:
            created_date = datetime.fromisoformat(metadata['created_at']).strftime("%Y-%m-%d %H:%M")
            st.sidebar.markdown(f"**Data generated:** {created_date}")

    # Add threshold legend
    st.sidebar.markdown("## Threshold Levels")
    st.sidebar.markdown("- **Conservative** (5%): Conservative approach - Light green")
    st.sidebar.markdown("- **Recommend Go** (10%): Standard recommendation - Medium green")
    st.sidebar.markdown("- **Aggressive** (25%): Aggressive strategy - Dark green")

    # Scenario selection
    scenario_options = {}
    for key, data in all_scenario_data.items():
        scenario_options[data['title']] = key

    st.sidebar.markdown("## Select Scenarios")
    selected_scenarios = st.sidebar.multiselect(
        "Choose scenarios to display:",
        options=list(scenario_options.keys()),
        default=list(scenario_options.keys())  # Show all by default
    )

    if not selected_scenarios:
        st.warning("Please select at least one scenario to display.")
        return

    # Display selected scenarios
    for scenario_title in selected_scenarios:
        scenario_key = scenario_options[scenario_title]
        data = all_scenario_data[scenario_key]

        try:
            with st.spinner(f"Rendering {scenario_title}..."):
                fig = create_heatmap_from_data(data, scenario_title)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Failed to render {scenario_title}")
        except Exception as e:
            logger.error(f"Error rendering scenario {scenario_title}: {e}")
            st.error(f"Error rendering {scenario_title}: {str(e)}")
            continue

    # Usage instructions
    st.markdown("---")
    st.markdown("**Performance Info:** This app uses pre-computed data for instant loading!")
    st.markdown("**To add new scenarios:** Run `python precalculate_data.py` and modify the scenarios list.")


if __name__ == "__main__":
    main()
st.markdown("**To run this app:** `streamlit run Game_Book_interactive.py`")