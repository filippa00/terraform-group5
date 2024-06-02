# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import joblib
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import random
import plotly.express as px
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from azure.ai.ml.entities import Model
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model

df = pd.read_csv('2016_CvH.csv')

# use less rows to take less time
df = df.iloc[:5000]
# Display the first few rows of the DataFrame
df.head(1)

print(f"The dataset includes {df.shape[0]} samples each having {df.shape[1]-1} features to help detect the cancerous tumors")
print(f"These features include the following {df.columns.to_numpy()[:-1]}")


# First drop redundant columns
df = df.drop(['Result','Commentaries','Moves'], axis=1)

df.info()

numeric_columns = df.select_dtypes(include=[np.number]).columns
print(f"Numeric columns of the dataset are: {numeric_columns}")

# Adding new features

# Function to perform some operation on columns and return a value
def white_time_min(row):
    try:
        hours_minute_part = row["White Clock"].split('.')[0]
        hours, minutes = map(int, hours_minute_part.split(':'))

        # Convert hours and minutes to minutes and add them together
        total_minutes = hours * 60 + minutes
        return total_minutes
    except:
        return None

def black_time_min(row):
    try:
        hours_minute_part = row["Black Clock"].split('.')[0]
        hours, minutes = map(int, hours_minute_part.split(':'))

        # Convert hours and minutes to minutes and add them together
        total_minutes = hours * 60 + minutes
        return total_minutes
    except Exception as e:
        print(e)
        return None

def start_time(row):
    try:
        hours, minutes, seconds = map(int, row["Time"].split(':'))
        return hours
    except Exception as e:
        print(e)
        return None

# Applying the function to create a new columns
df['white_time_min'] = df.apply(white_time_min, axis=1)

df['black_time_min'] = df.apply(black_time_min, axis=1)
df['state_time'] = df.apply(start_time, axis=1)

# How to deal with null values
null_value_action = 'replace_median' # 'drop'
number_of_null_values = df.isnull().sum().sum()
print(f"There are {number_of_null_values} null values in the dataset")
if number_of_null_values > 0:
    if null_value_action == 'drop':
        df.dropna(axis=0, inplace=True)
        print(f"removed rows with null values")
    elif null_value_action == 'replace_median':
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        print(f"filled null values with median of the column")
else:
    print("No null values. No action needed")
number_of_duplicate_rows = df.duplicated().sum()
print(f"There are {number_of_duplicate_rows} duplicate rows in the dataset")


df.head(1)

print(f"Target column type is {df['Result-Winner'].dtypes} with following values: {df['Result-Winner'].unique()}")


x_v = df['Result-Winner'].value_counts().index
y_v = df['Result-Winner'].value_counts().values

df_v = pd.DataFrame({
    'target': x_v,
    'count': y_v
})

labels = df['Result-Winner'].unique()
values = df_v['count']

plt.pie(values, labels=labels, autopct=lambda p: f'{p:.1f}%\n({int(p * sum(values) / 100)})')
plt.axis('equal')
plt.show()

# We should convert target column to numeric too.
unique_labels = df['Result-Winner'].unique()
label_mapping = {label: index for index, label in enumerate(unique_labels)}

df['Result-Winner'] = df['Result-Winner'].map(label_mapping)
numeric_columns = df.select_dtypes(include=[np.number]).columns
df = df[numeric_columns]

y = df['Result-Winner']
X = df.drop('Result-Winner', axis=1)
# We want to use numeric values to predict since columns such as name of players are not important in our model.


print("Features extraction using mutual information scores")


print("Feature extraction using feature importance")
model = RandomForestRegressor()
model.fit(X, y)
importances = model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
sorted_feature_names = X.columns[sorted_indices]

for i, feature_name in enumerate(sorted_feature_names):
    print(f"{i+1}. {feature_name}:\t {importances[sorted_indices[i]]}")


print(f"Feature extraction using correlation")
correlation = df.corr()['Result-Winner']
correlation_sorted = correlation.sort_values(ascending=False)
print(correlation_sorted[:11])


numeric_features = ['White Elo', 'Black Elo', 'White RD', 'Black RD', 'PlyCount', 'white_time_min', 'black_time_min', 'state_time']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# add the transformer to the pre-processor variable
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', numeric_transformer, numeric_features)
    ])

# create a pipeline and append the Support Vector Classifier classifier
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', SVC())])

param_grid = [
    {
        'classifier': [SVC()],
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf']
    },
    {
        'classifier': [GaussianNB()]
    },
    {
        'classifier': [KNeighborsClassifier()],
        'classifier__n_neighbors': [3, 5, 7]
    },
    {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 5, 10]
    },
    {
        'classifier': [DecisionTreeClassifier()],
        'classifier__max_depth': [None, 5, 10]
    },
    {
        'classifier': [LogisticRegressionCV()],
        'classifier__solver': ['liblinear', 'lbfgs']
    },
    {
        'classifier': [MLPClassifier(random_state=42)],
        'classifier__hidden_layer_sizes': [(10,), (10, 5)],
        'classifier__activation': ['relu', 'tanh'],
    }
]

grid_search = GridSearchCV(pipeline, param_grid, cv=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

print(f"Size of train data {X_train.shape}, {y_train.shape} and test data {X_test.shape}, {y_test.shape}")

grid_search.fit(X_train, y_train)
# Access the CV results
cv_results = grid_search.cv_results_

# Loop through each parameter setting and print the accuracy
for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
    print("Mean Accuracy:", mean_score, "Parameters:", params)


best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Extract the model name
model_name = best_model.named_steps['classifier'].__class__.__name__

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
accuracy = best_model.score(X_test, y_test)

print("Best Model:", model_name)
print("Best Parameters:", best_params)
print("Test Accuracy:", accuracy)
print("Precision Score: ", precision_score(y_test, y_pred,average='macro'))
print("Recall Score: ", recall_score(y_test, y_pred,average='macro'))
print("F1 Score: ", f1_score(y_test, y_pred,average='macro'))


hyperparameters = best_model.get_params()

# Print the hyperparameters
for param, value in hyperparameters.items():
    print(param, "=", value)


print("Confusion matrix")

k = len(df['Result-Winner'].unique())

initials = ['White' 'Black' 'Draw'] # static list to properly display the labels without overlap

#displaying confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

majority_class = y_train.value_counts().idxmax()

# Calculate the baseline accuracy
baseline_accuracy = y_test.value_counts()[majority_class] / len(y_test)

print("Baseline Accuracy:", baseline_accuracy)
joblib.dump(model, "trained_model.pkl")

