import os
import pandas as pd
import numpy as np

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# List all CSV files in the directory
csv_files = [f for f in os.listdir(script_dir) if f.endswith(".csv")]

# List to store score columns
score_list = []

for file in csv_files:
    file_path = os.path.join(script_dir, file)
    try:
        df = pd.read_csv(file_path, usecols=["score"])  # Import only 'score' column
        score_list.append(df)  # Append to list
        print(f"Imported 'score' column from {file}")
    except ValueError:
        print(f"Skipping {file} (no 'score' column)")

# Concatenate all score columns into one DataFrame
if score_list:
    combined_scores = pd.concat(score_list, ignore_index=True)
    print("\nCombined 'score' DataFrame:")
    print(combined_scores.head())

    # Loop through every row and check if it is zero
    print("\nChecking each row for score == 0:")
    game_scores = []
    new_zero = 1
    previous_value = 0
    for idx, row in combined_scores.iterrows():
        if row['score'] == 0 and new_zero == 0:
            game_scores.append(previous_value)
            new_zero = 1
        elif row['score'] != 0:
            new_zero = 0
            previous_value = row['score']
            # Additional manipulations can be added here
    print(f"\nGame scores: {game_scores}")
    differences = [abs(int(str(score).split('.')[0]) - int(str(score).split('.')[1]))
               if '.' in str(score) else 0  # Handle cases where there's no decimal part
               for score in game_scores]

# Calculate standard deviation
    std_dev = np.std(differences, ddof=1)  # Sample standard deviation

    print("Absolute differences:", differences)
    print("Standard deviation of differences:", std_dev)
