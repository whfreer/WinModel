import numpy as np
import pickle
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your decision function
try:
    from PUNT_FG_GO import go_for_it_decision
except ImportError as e:
    logger.error(f"Failed to import go_for_it_decision: {e}")
    raise


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


def calculate_scenario_data(time, score_diff, down, kicker_range, thresholds):
    """Calculate all heatmap data for a given scenario"""

    print(f"Calculating data for: Time={time}, Score_diff={score_diff}, Down={down}, Kicker_range={kicker_range}")

    yards_to_goals = np.arange(1, 100)  # absolute yardline (1..99)
    yards_to_go = np.arange(1, 21)  # yards to go (1..20)
    max_shift = yards_to_go[-1]
    full_length = len(yards_to_goals) + max_shift

    # Define threshold levels with names
    threshold_levels = {
        0.05: "Conservative",
        0.10: "Recommend Go",
        0.25: "Aggressive"
    }

    # Storage for all data
    scenario_data = {
        'go_for_it_score_grid': np.full((len(yards_to_go), full_length), np.nan),
        'cell_raw_probs': {},
        'cell_threshold_info': {},
        'yards_to_go': yards_to_go,
        'yards_to_goals': yards_to_goals,
        'max_shift': max_shift,
        'parameters': {
            'time': time,
            'score_diff': score_diff,
            'down': down,
            'kicker_range': kicker_range,
            'thresholds': thresholds
        }
    }

    total_calculations = len(yards_to_go) * len(yards_to_goals)
    completed = 0

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

                    # Store data
                    cell_key = (int(distance), yards_between)
                    scenario_data['cell_raw_probs'][cell_key] = raw_acc
                    scenario_data['cell_threshold_info'][cell_key] = threshold_recs

                    # Determine final recommendation
                    if go_count >= 3:  # All 3 thresholds say go
                        scenario_data['go_for_it_score_grid'][i, shifted_col] = 4  # Must Go
                    elif go_count >= 2:  # 2 out of 3 say go
                        scenario_data['go_for_it_score_grid'][i, shifted_col] = 3  # Recommend Go
                    elif go_count >= 1:  # Only 50-50 threshold says go
                        scenario_data['go_for_it_score_grid'][i, shifted_col] = 2  # 50-50 Go
                    elif fallback_counts['Field Goal'] > fallback_counts['Punt']:
                        scenario_data['go_for_it_score_grid'][i, shifted_col] = 1  # FG
                    else:
                        scenario_data['go_for_it_score_grid'][i, shifted_col] = 0  # Punt

            completed += 1
            if completed % 100 == 0:
                print(f"Progress: {completed}/{total_calculations} ({completed / total_calculations * 100:.1f}%)")

    return scenario_data


def main():
    """Pre-calculate data for all scenarios"""

    # Define all scenarios you want to pre-calculate
    scenarios = [
        {
            "time": 2700,
            "score_diff": 17,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "1Q/2Q up 17+"
        },
        {
            "time": 2700,
            "score_diff": 10,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "1Q/2Q up 8-16"
        },
        {
            "time": 2700,
            "score_diff": 0,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "1Q/2Q -7 to +7"
        },
        {
            "time": 2700,
            "score_diff": -10,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "1Q/2Q down 8-16"
        },
        {
            "time": 2700,
            "score_diff": -17,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "1Q/2Q down 17+"
        },
        {
            "time": 1400,
            "score_diff": 17,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "3Q, up 17+"
        },
        {
            "time": 1400,
            "score_diff": 14,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "3Q, up 11-16"
        },
        {
            "time": 1400,
            "score_diff": 10,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "3Q, up 9-10"
        },
        {
            "time": 1400,
            "score_diff": 6,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "3Q, up 4-8"
        },
        {
            "time": 1400,
            "score_diff": 2,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "3Q, up 1-3"
        },
        {
            "time": 1400,
            "score_diff": 0,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "3Q, tied"
        },
        {
            "time": 1400,
            "score_diff": -2,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "3Q, down 1-3"
        },
        {
            "time": 1400,
            "score_diff": -6,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "3Q, down 4-8"
        },
        {
            "time": 1400,
            "score_diff": -10,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "3Q, down 9-10"
        },
        {
            "time": 1400,
            "score_diff": -14,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "3Q, down 11-16"
        },
        {
            "time": 1400,
            "score_diff": -17,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "3Q, down 17+"
        },
        {
            "time": 500,
            "score_diff": 17,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "4Q up 17+"
        },
        {
            "time": 500,
            "score_diff": 14,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "4Q up 11-16"
        },
        {
            "time": 500,
            "score_diff": 10,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "4Q up 9-10"
        },
        {
            "time": 500,
            "score_diff": 6,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "4Q up 4-8"
        },
        {
            "time": 500,
            "score_diff": 2,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "4Q up 1-3"
        },
        {
            "time": 500,
            "score_diff": 0,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "4Q tied"
        },
        {
            "time": 500,
            "score_diff": -2,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "4Q down 1-3"
        },
        {
            "time": 500,
            "score_diff": -6,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "4Q down 4-8"
        },
        {
            "time": 500,
            "score_diff": -10,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "4Q down 9-10"
        },
        {
            "time": 500,
            "score_diff": -14,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "4Q down 11-16"
        },
        {
            "time": 500,
            "score_diff": -17,
            "down": 4,
            "kicker_range": 55,
            "thresholds": [0.05, 0.10, 0.25],
            "title": "4Q down 17+"
        },
    ]

    all_scenario_data = {}

    print(f"Starting pre-calculation of {len(scenarios)} scenarios...")
    print(f"Started at: {datetime.now()}")

    for i, scenario in enumerate(scenarios):
        print(f"\n--- Scenario {i + 1}/{len(scenarios)}: {scenario['title']} ---")

        try:
            # Create a unique key for this scenario
            scenario_key = f"{scenario['time']}_{scenario['score_diff']}_{scenario['down']}_{scenario['kicker_range']}"

            # Calculate data
            data = calculate_scenario_data(
                scenario['time'],
                scenario['score_diff'],
                scenario['down'],
                scenario['kicker_range'],
                scenario['thresholds']
            )

            # Add title to data
            data['title'] = scenario['title']

            # Store in master dictionary
            all_scenario_data[scenario_key] = data

            print(f"✅ Completed scenario {i + 1}: {scenario['title']}")

        except Exception as e:
            print(f"❌ Error in scenario {i + 1}: {e}")
            logger.error(f"Error calculating scenario {i + 1}: {e}")
            continue

    # Save all data
    print(f"\nSaving data...")

    # Save as pickle (most efficient for numpy arrays)
    with open('heatmap_data.pkl', 'wb') as f:
        pickle.dump(all_scenario_data, f)

    # Also save scenario metadata as JSON for easy reference
    metadata = {
        'scenarios': {key: data['parameters'] for key, data in all_scenario_data.items()},
        'titles': {key: data['title'] for key, data in all_scenario_data.items()},
        'created_at': datetime.now().isoformat(),
        'total_scenarios': len(all_scenario_data)
    }

    with open('scenario_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ All data saved!")
    print(f"✅ Created heatmap_data.pkl ({len(all_scenario_data)} scenarios)")
    print(f"✅ Created scenario_metadata.json")
    print(f"Finished at: {datetime.now()}")

    # Print summary
    print(f"\n--- SUMMARY ---")
    for key, data in all_scenario_data.items():
        params = data['parameters']
        print(f"{key}: {data['title']} (Time: {params['time']}, Score: {params['score_diff']})")


if __name__ == "__main__":
    main()