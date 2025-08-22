import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Load the CSV file
file_path = '2024_NoIDNums.csv'  # Make sure this path is correct
df = pd.read_csv(file_path, low_memory=False)

# Encode 'offensive_win_loss' as 0 (Loss) and 1 (Win)
df['offensive_win_loss'] = df['offensive_win_loss'].map({'L': 0, 'W': 1})

# Convert 'clock' column to seconds (from mm:ss format)
def convert_clock_to_seconds(clock_str):
    try:
        minutes, seconds = map(int, clock_str.split(':'))
        return minutes * 60 + seconds
    except:
        return np.nan  # In case of invalid format

df['clock'] = df['clock'].apply(convert_clock_to_seconds)

# Drop non-numeric columns except 'clock' and 'offensive_win_loss'
df_numeric = df.select_dtypes(include=[np.number])

# Drop columns with all NaN values (completely empty columns)
df_numeric = df_numeric.dropna(axis=1, how='all')

# Impute missing values with the column mean
imputer = SimpleImputer(strategy='mean')
df_numeric_imputed = imputer.fit_transform(df_numeric)

# Convert the imputed array back to a DataFrame with original column names
df_numeric_imputed = pd.DataFrame(df_numeric_imputed, columns=df_numeric.columns)
df_numeric_imputed = df_numeric_imputed.drop(columns=['offensive_win_loss'])
# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric_imputed)

# Perform PCA
pca = PCA()
pca.fit(df_scaled)

# Scree plot
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_, marker='o', linestyle='-')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(np.arange(1, len(pca.explained_variance_ratio_) + 1))
plt.grid(True)
plt.tight_layout()
plt.show()
print(f"Number of principal components: {pca.n_components_}")
X = 70  # Change this to your desired number of components
cumulative_variance = np.sum(pca.explained_variance_ratio_[:X])
print(f"Cumulative explained variance by the first {X} components: {cumulative_variance:.4f}")

