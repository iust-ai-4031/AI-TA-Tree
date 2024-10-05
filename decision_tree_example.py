# %% [markdown]
# # Multi-Node Categorical Decision Tree Classifier Example

# %% [markdown]
# ## Importing Libraries

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from decision_tree import MultiNodeCategoricalDecisionTree

# %% [markdown]
# ## Loading the Dataset
# 
# TODO: Replace this section with code to load your actual dataset.
# 
# In this section, you should:
# 1. Load your dataset from a file (CSV, Excel, etc.) or a database
# 2. Display basic information about the dataset (shape, columns, etc.)
# 3. Show the first few rows of the data

# %%
# TODO: Load the actual dataset
def load_dataset():
    # This is a placeholder. Replace with actual data loading code.
    data = pd.DataFrame({
        'feature1': ['A', 'B', 'C', 'A', 'B'] * 20,
        'feature2': ['X', 'Y', 'Z', 'X', 'Y'] * 20,
        'feature3': ['P', 'Q', 'R', 'P', 'Q'] * 20,
        'target': [0, 1, 0, 1, 0] * 20
    })
    return data

data = load_dataset()
print(data.head())

# %% [markdown]
# ## Exploratory Data Analysis (EDA)
# 
# TODO: Perform exploratory data analysis on your dataset.
# 
# In this section, you should:
# 1. Analyze the distribution of features and target variable
# 2. Check for missing values and outliers
# 3. Visualize relationships between features and the target variable
# 4. Identify any patterns or correlations in the data

# %%
# TODO: Add your EDA code here
# Example:
# print(data.describe())
# print(data.isnull().sum())
# Add visualizations using matplotlib or seaborn

# %% [markdown]
# ## Preprocessing
# 
# TODO: Preprocess your data to prepare it for the decision tree model.
# 
# In this section, you should:
# 1. Handle missing values (if any)
# 2. Encode categorical variables
# 3. Split the data into features (X) and target (y)
# 4. Split the data into training and testing sets
# 
# Note: You should encode your data before splitting it into features and target, 
# because the decision tree classifier works with categorical data.
# 
# Example of encoding data:
# 

# %%
"""
TODO: Add your preprocessing code here
Note : you should encode your data before split it into features and target ,
because decision tree classifier just work with categorical data

example of encoding data :

le = LabelEncoder()
X = data.drop('target', axis=1)
y = data['target']

Encode categorical features
Features need to be encoded using OrdinalEncoder search about this function in sklearn library

"""
for column in X.columns:
    X[column] = le.fit_transform(X[column])

# Split the data
X_train, X_test, y_train, y_test = # TODO : Split dataset using sklearn library ,naturally you should search about this function
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# %% [markdown]
# ## Training the Model

# %%
# Initialize and train the model
model = MultiNodeCategoricalDecisionTree(max_depth=5, min_samples_split=2)
model.fit(X_train.values, y_train.values)

# %% [markdown]
# ## Evaluating the Model

# %%
# Make predictions
y_pred = model.predict(X_test.values)

# Ensure y_pred is in the same format as y_test
y_pred = np.array(y_pred).astype(int)
y_test = np.array(y_test).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# %% [markdown]
# ## Feature Importances

# %%
# Display feature importances
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importances)

# %% [markdown]
# ## Preparing Submission

# %%
# TODO: Replace this with your actual test data loading code
def load_test_data():
    # This is a placeholder. Replace with actual test data loading code.
    test_data = pd.DataFrame({
        'feature1': ['A', 'B', 'C', 'A', 'B'] * 10,
        'feature2': ['X', 'Y', 'Z', 'X', 'Y'] * 10,
        'feature3': ['P', 'Q', 'R', 'P', 'Q'] * 10,
    })
    return test_data

test_data = load_test_data()

# Preprocess test data
for column in test_data.columns:
    test_data[column] = le.fit_transform(test_data[column])

# Make predictions on test data
test_predictions = model.predict(test_data.values)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': range(len(test_predictions)),
    'predicted_target': test_predictions
})

print(submission.head())

# Save submission to CSV
submission.to_csv('submission.csv', index=False)
print("Submission saved to 'submission.csv'")


