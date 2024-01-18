#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:01:27 2024

@author: vedagrawal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Function to get the normalized composite score for a given player
def get_normalized_score(player_name, dataframe):
    player_score = dataframe[dataframe['Player'].str.contains(player_name, case=False, na=False)]['Normalized_Composite_Score_per_Min']
    if not player_score.empty:
        return player_score.iloc[0]
    else:
        return None

# Load data from multiple seasons
seasons = ["2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023"]
all_seasons_data = []

for season in seasons:
    season_data = pd.read_csv(f"per-game-{season}.csv")
    season_data['Season'] = season  # Optionally add a season identifier
    all_seasons_data.append(season_data)

# Combine all seasons into a single DataFrame
df_all_seasons = pd.concat(all_seasons_data)


# Assign weights to each stat
weights = {
    'PTS': 1,
    'AST': 0.75,
    'STL': 0.3,
    'BLK': 0.25,
    'TOV': -1,
    'PF': -0.5
}

# Calculate the composite score per minute
df_all_seasons['Composite_Score_per_Min'] = (
    weights['PTS'] * df_all_seasons['PTS'] +
    weights['AST'] * df_all_seasons['AST'] +
    weights['STL'] * df_all_seasons['STL'] +
    weights['BLK'] * df_all_seasons['BLK'] +
    weights['TOV'] * df_all_seasons['TOV'] +
    weights['PF'] * df_all_seasons['PF']
) / df_all_seasons['MP']

# Replace infinite or NaN values resulting from division by zero with the column mean
df_all_seasons['Composite_Score_per_Min'].replace([np.inf, -np.inf], np.nan, inplace=True)
df_all_seasons['Composite_Score_per_Min'].fillna(df_all_seasons['Composite_Score_per_Min'].mean(), inplace=True)

# Normalize the composite scores to be between 0 and 1
df_all_seasons['Normalized_Composite_Score_per_Min'] = (
    df_all_seasons['Composite_Score_per_Min'] - df_all_seasons['Composite_Score_per_Min'].min()
) / (df_all_seasons['Composite_Score_per_Min'].max() - df_all_seasons['Composite_Score_per_Min'].min())

# Select features for the model
X = df_all_seasons[['PTS', 'AST', 'STL', 'BLK', 'TOV', 'PF']]
y = df_all_seasons['Composite_Score_per_Min']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Predict the composite score per minute on the test set
y_pred = rf_regressor.predict(X_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error of the predictions: {mse}')

# Prompt the user to enter a player's name
player_name = input("Enter a player's name to get their normalized composite score: ")

# Get and print the normalized composite score for the entered player
player_normalized_score = get_normalized_score(player_name, df_all_seasons)
if player_normalized_score is not None:
    print(f"{player_name}'s Normalized Composite Score per Minute: {player_normalized_score}")
else:
    print(f"Player '{player_name}' not found in the dataset.")

# Histogram of the normalized composite scores
plt.figure(figsize=(10, 6))
sns.histplot(df_all_seasons['Normalized_Composite_Score_per_Min'], bins=30, kde=False, color='skyblue')
plt.title('Distribution of Normalized Composite Scores per Minute for NBA Players (All Seasons)')
plt.xlabel('Normalized Composite Score per Minute')
plt.ylabel('Number of Players')
plt.show()


