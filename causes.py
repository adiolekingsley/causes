import pandas as pd

import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

# Load the dataset into a pandas DataFrame

data = pd.DataFrame({

    'Year': [2020, 2021, 2022],

    'Number of Road Accidents': [11721, 11641, 2826],

    'Injuries': [6266, 6566, 1578],

    'Fatalities': [3328, 3581, 814],

    'Cause': ['OverSpeeding', 'OverSpeeding, loss of control, tyre blowouts', 'OverSpeeding, loss of control, tyre blowouts']

})

# Split the data into features and target

X = data.iloc[:, :-1]

y = data.iloc[:, -1]

# Encode the target variable

le = LabelEncoder()

y = le.fit_transform(y)

# Create a decision tree classifier and fit it to the data

tree = DecisionTreeClassifier(criterion='entropy')

tree.fit(X, y)

# Use the model to make predictions on new data

new_data = pd.DataFrame({

    'Year': [2023],

    'Number of Road Accidents': [15000],

    'Injuries': [8000],

    'Fatalities': [4000],

})

new_data_pred = tree.predict(new_data)

# Decode the predicted target variable

new_data_pred = le.inverse_transform(new_data_pred)

print('Predicted cause of road accident:', new_data_pred)

