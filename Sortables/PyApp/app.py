import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Setup
st.set_page_config(page_title="4th Down Conversions", layout="centered")

# Title
st.title("4th Down Conversion % vs Distance")

# Load data
@st.cache_data
def load_data(path):
    columns = ['quarter', 'down', 'clock', 'def_score', 'off_score', 'distance',
               'off_timeouts_remaining', 'def_timeouts_remaining', 'yards_to_goal_line',
               'offensive_win_loss', 'first_down_gained', 'home_team', 'away_team', 'special_teams_type']
    return pd.read_csv(path, usecols=columns)

# Data path
csv_path = '2024_fbs_season (1).csv'

# Load once
df = load_data(csv_path)

# Get team list
teams = sorted(set(df['home_team'].dropna().unique()) | set(df['away_team'].dropna().unique()))

# User selects team
team_abbr = st.selectbox("Select a team", teams, index=0)

# Smoothing window size
window_size = st.slider("Smoothing Window Size", min_value=1, max_value=15, value=5)

# Filter + plot logic
def plot_team_4th_down_conversions(team_abbr, df, window_size):
    df = df[(df['home_team'] == team_abbr) | (df['away_team'] == team_abbr)]
    df_4th = df[(df['down'] == 4) & (df['special_teams_type'].isna())].dropna(subset=['distance'])
    df_4th['distance'] = df_4th['distance'].round(1)
    df_4th['conversion'] = df_4th['first_down_gained'].fillna(0).astype(int)

    grouped = df_4th.groupby('distance').agg(
        conversions=('conversion', 'sum'),
        attempts=('conversion', 'count')
    )
    grouped['conversion_pct'] = grouped['conversions'] / grouped['attempts']
    grouped['smoothed_pct'] = grouped['conversion_pct'].rolling(window=window_size, center=True).mean()

    # Plot with matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grouped.index, grouped['conversion_pct'], marker='o', linestyle=':', color='gray', label='Raw %')
    ax.plot(grouped.index, grouped['smoothed_pct'], color='blue', linewidth=2, label='Smoothed %')
    ax.plot(grouped.index, grouped['attempts'] / grouped['attempts'].max(), linestyle='--', color='lightgreen',
            label='Relative Attempt Count')

    ax.set_title(f'{team_abbr} 4th Down Conversion %')
    ax.set_xlabel('Distance to Go (yards)')
    ax.set_ylabel('Conversion Percentage')
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# Display plot
plot_team_4th_down_conversions(team_abbr, df, window_size)
