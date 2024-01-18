#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 17:06:58 2023

@author: vedagrawal
"""

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data from multiple seasons
seasons = ["2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023"]
all_seasons_data = []

for season in seasons:
    season_data = pd.read_csv(f"per-game-{season}.csv")
    season_data['Season'] = season  # Optionally add a season identifier
    all_seasons_data.append(season_data)

# Combine all seasons into a single DataFrame
df_all_seasons = pd.concat(all_seasons_data)

# Select features for multicollinearity check
features = df_all_seasons[['FGA', 'FG', 'PTS', 'GS', 'TOV', 'PF']]
features_with_constant = sm.add_constant(features)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data['feature'] = features_with_constant.columns
vif_data['VIF'] = [variance_inflation_factor(features_with_constant.values, i) for i in range(features_with_constant.shape[1])]

print("Variance Inflation Factor for each feature:")
print(vif_data)

#Correlation Matrix
df_all_seasons_numeric = df_all_seasons.drop(['Player', 'Pos', 'Tm', 'Season', 'Player-additional'], axis=1)
corr = df_all_seasons_numeric.corr()
# Create a heatmap
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Feature selection based on VIF results
X = df_all_seasons[['PTS', 'GS', 'TOV', 'PF']]  # Predictor variables
y = df_all_seasons['MP']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

# Predict on the test set
y_pred = linear_reg_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("R-squared:", r2)



# Prompt the user to enter a player's name
player_name = input("\nEnter a player's name to get their predicted minutes: ")

# Find the row for the entered player and extract their stats for the model's features
player_stats = df_all_seasons[df_all_seasons['Player'].str.contains(player_name, case=False, na=False)][['PTS', 'GS', 'TOV', 'PF']]

# Predict their optimal minutes
predicted_minutes_player = linear_reg_model.predict(player_stats)
optimal_minutes_player = predicted_minutes_player[0] if not player_stats.empty else "Player not found in the dataset."
print(f"Predicted optimal minutes for {player_name}:", optimal_minutes_player)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Actual Minutes Played')
plt.ylabel('Predicted Minutes Played')
plt.title('Actual vs. Predicted Minutes Played')
plt.show()

