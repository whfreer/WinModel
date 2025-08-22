import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configure logging for error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import the decision function with error handling
try:
    from PUNT_FG_GO import go_for_it_decision
except ImportError as e:
    logger.error(f"Failed to import go_for_it_decision: {e}")
    st.error("Error: Could not import PUNT_FG_GO module. Please ensure it's available.")
    st.stop()

st.set_page_config(page_title="4th Down Decision Heatmaps", layout="wide")
st.title("Multiple 4th Down Decision Heatmaps")


def validate_inputs(time, score_diff, down, kicker_range, thresholds):
    """Validate input parameters"""
    errors = []

    if not isinstance(time, (int, float)) or time < 0:
        errors.append("Time must be a positive number")
    if not isinstance(score_diff, (int, float)):
        errors.append("Score difference must be a number")
    if not isinstance(down, int) or down < 1 or down > 4:
        errors.append("Down must be an integer between 1 and 4")
    if not isinstance(kicker_range, (int, float)) or kicker_range < 0:
        errors.append("Kicker range must be a positive number")
    if not isinstance(thresholds, list) or len(thresholds) == 0:
        errors.append("Thresholds must be a non-empty list")

    return errors


def safe_decision_call(time, score_diff, down, distance, ytg, kicker_range, threshold):
    """Safely call the go_for_it_decision function with error handling"""
    try:
        probs, rec = go_for_it_decision(time, score_diff, down, distance, ytg, kicker_range, threshold)

        # Validate return values
        if not isinstance(probs, dict):
            raise ValueError("go_for_it_decision should return probabilities as a dictionary")
        if not isinstance(rec, str):
            raise ValueError("go_for_it_decision should return recommendation as a string")

        # Ensure all required keys exist
        required_keys = ['Punt', 'Field Goal', 'Go for It']
        for key in required_keys:
            if key not in probs:
                probs[key] = 0.0
                logger.warning(f"Missing probability key '{key}', setting to 0.0")

        return probs, rec

    except Exception as e:
        logger.error(f"Error in go_for_it_decision: {e}")
        # Return default values
        return {'Punt': 0.0, 'Field Goal': 0.0, 'Go for It': 0.0}, 'Punt'


def make_heatmap(time, score_diff, down, kicker_range, thresholds, title=""):
    """Build one heatmap figure with improved error handling and hover alignment"""

    # Validate inputs
    validation_errors = validate_inputs(time, score_diff, down, kicker_range, thresholds)
    if validation_errors:
        st.error(f"Input validation errors: {'; '.join(validation_errors)}")
        return None

    try:
        yards_to_goals = np.arange(1, 100)  # absolute yardline (1..99)
        yards_to_go = np.arange(1, 21)  # yards to go (1..20)
        max_shift = yards_to_go[-1]
        full_length = len(yards_to_goals) + max_shift

        # Define threshold levels with names
        threshold_levels = {
            0.025: "50-50",
            0.05: "Recommend Go",
            0.10: "Must Go"
        }

        # working grid + dictionary to store raw probs
        go_for_it_score_grid = np.full((len(yards_to_go), full_length), np.nan)
        cell_raw_probs = {}
        cell_threshold_info = {}

        for i, distance in enumerate(yards_to_go):
            for j, ytg in enumerate(yards_to_goals):
                if ytg >= distance:
                    shifted_col = j - distance + max_shift
                    yards_between = int(ytg - distance)

                    if yards_between <= 90:
                        go_count = 0
                        fallback_counts = {'Punt': 0, 'Field Goal': 0}
                        raw_acc = {'Punt': 0.0, 'Field Goal': 0.0, 'Go for It': 0.0}
                        threshold_recs = []

                        for t in thresholds:
                            probs, rec = safe_decision_call(time, score_diff, down, distance, ytg, kicker_range, t)

                            # Accumulate probabilities
                            for k in raw_acc:
                                if k in probs:
                                    raw_acc[k] += probs[k] / len(thresholds)

                            # Track recommendations
                            threshold_name = threshold_levels.get(t, f"{t:.3f}")
                            threshold_recs.append(f"{threshold_name}: {rec}")

                            if rec == 'Go for It':
                                go_count += 1
                            else:
                                fallback_counts[rec] += 1

                        # Store data with correct indexing for hover
                        cell_key = (int(distance), yards_between)
                        cell_raw_probs[cell_key] = raw_acc
                        cell_threshold_info[cell_key] = threshold_recs

                        # Determine final recommendation with enhanced thresholds (5 levels)
                        if go_count >= 3:  # All 3 thresholds say go
                            go_for_it_score_grid[i, shifted_col] = 4  # Must Go (dark green)
                        elif go_count >= 2:  # 2 out of 3 say go
                            go_for_it_score_grid[i, shifted_col] = 3  # Recommend Go (medium green)
                        elif go_count >= 1:  # Only 50-50 threshold says go
                            go_for_it_score_grid[i, shifted_col] = 2  # 50-50 Go (light green)
                        elif fallback_counts['Field Goal'] > fallback_counts['Punt']:
                            go_for_it_score_grid[i, shifted_col] = 1  # FG (yellow)
                        else:
                            go_for_it_score_grid[i, shifted_col] = 0  # Punt (red)

        # x-axis mapping - FIXED for proper hover alignment
        x_values = np.arange(-max_shift, len(yards_to_goals))
        valid_mask = (x_values >= 0) & (x_values <= 89)
        x_values_filtered = x_values[valid_mask]
        heat = go_for_it_score_grid[:, valid_mask]
        nrows, ncols = heat.shape

        # Create hover text with FIXED indexing - offset by -1 to correct alignment
        hover_text = np.full(heat.shape, "", dtype=object)
        for i, distance in enumerate(yards_to_go):
            for j in range(ncols):
                # FIXED: The key issue is we need to offset by -1 to align with visual
                # The visual display is shifted by one position from our data storage
                yards_between = int(x_values_filtered[j])

                # Try the current position first, then adjacent positions if needed
                possible_keys = [
                    (int(distance), yards_between),
                    (int(distance), yards_between - 1),  # Try one position left
                    (int(distance), yards_between + 1),  # Try one position right
                ]

                found_data = False
                for key in possible_keys:
                    if key in cell_raw_probs and not np.isnan(heat[i, j]):
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

        # Enhanced discrete colors for 5 levels (added third green for 50-50)
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

        # Split into two halves (Opponent vs Own territory)
        midpoint = ncols // 2
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                "Recommended 4th Down Decision (Opponent Territory)",
                "Recommended 4th Down Decision (Own Territory)"
            )
        )

        # Use simple integer indices for x-axis to ensure proper alignment
        x_indices = list(range(ncols))

        # Opponent territory (first half)
        fig.add_trace(go.Heatmap(
            z=heat[:, :midpoint],
            x=x_indices[:midpoint],
            y=yards_to_go,
            text=hover_text[:, :midpoint],
            hovertemplate='%{text}<extra></extra>',
            colorscale=colorscale,
            zmin=0, zmax=4,
            showscale=True,
            colorbar=dict(
                tickvals=[0, 1, 2, 3, 4],
                ticktext=["Punt", "Field Goal", "50-50 Go", "Recommend Go", "Must Go"],
                title="Decision",
                len=0.8
            )
        ), row=1, col=1)

        # Own territory (second half)
        fig.add_trace(go.Heatmap(
            z=heat[:, midpoint:],
            x=x_indices[:ncols - midpoint],  # Reset indices for second half
            y=yards_to_go,
            text=hover_text[:, midpoint:],
            hovertemplate='%{text}<extra></extra>',
            colorscale=colorscale,
            zmin=0, zmax=4,
            showscale=False
        ), row=2, col=1)

        # Update axes with custom tick labels showing actual yard values
        fig.update_yaxes(title="Yards to Go", autorange="reversed", row=1, col=1)
        fig.update_yaxes(title="Yards to Go", autorange="reversed", row=2, col=1)

        # Custom x-axis labels for opponent territory
        opponent_ticks = list(range(midpoint))
        opponent_labels = [str(x_values_filtered[i]) for i in range(midpoint)]
        fig.update_xaxes(
            title="Yards Between First Down Marker and Goal Line",
            tickvals=opponent_ticks,
            ticktext=opponent_labels,
            row=1, col=1
        )

        # Custom x-axis labels for own territory
        own_ticks = list(range(ncols - midpoint))
        own_labels = [str(x_values_filtered[i + midpoint]) for i in range(ncols - midpoint)]
        fig.update_xaxes(
            title="Yards Between First Down Marker and Goal Line",
            tickvals=own_ticks,
            ticktext=own_labels,
            row=2, col=1
        )

        fig.update_layout(
            title=f"{title}  |  Score Diff: {score_diff}  |  Time: {time}s",
            margin=dict(t=100, b=50, l=60, r=140),
            hoverlabel=dict(align="left", bgcolor="white", font_size=12),
            autosize=True,
            height=800
        )

        # Enhanced gridlines
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


# -----------------------------
# Example scenarios with error handling
# -----------------------------
try:
    scenarios = [
        {
            "time": 2000,
            "score_diff": 6,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.025, 0.05, 0.10],  # Updated thresholds: 50-50, Recommend Go, Must Go
            "title": "Scenario 1: Late Game, Up by 6"
        },
        {
            "time": 1200,
            "score_diff": 3,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.025, 0.05, 0.10],
            "title": "Scenario 2: Mid-Game, Up by 3"
        },
        {
            "time": 1600,
            "score_diff": -3,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.025, 0.05, 0.10],
            "title": "Scenario 3: Late Game, Down by 3"
        }
    ]

    # Add threshold legend
    st.sidebar.markdown("## Threshold Levels")
    st.sidebar.markdown("- **50-50** (2.5%): Marginal decision - Very light green")
    st.sidebar.markdown("- **Recommend Go** (5%): Strong recommendation - Medium green")
    st.sidebar.markdown("- **Must Go** (10%): Critical situation - Dark green")

    for i, sc in enumerate(scenarios):
        try:
            with st.spinner(f"Generating {sc['title']}..."):
                fig = make_heatmap(**sc)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Failed to generate {sc['title']}")
        except Exception as e:
            logger.error(f"Error processing scenario {i + 1}: {e}")
            st.error(f"Error processing {sc['title']}: {str(e)}")
            continue

except Exception as e:
    logger.error(f"Critical error in main execution: {e}")
    st.error(f"Critical error: {str(e)}")

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