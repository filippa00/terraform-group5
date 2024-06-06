# Imports
import numpy as np
import pandas as pd
import argparse
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Train and log a model on Azure ML")
    parser.add_argument('--data', type=str, required=True, help='Path to the input data CSV file')
    parser.add_argument('--num-threads', type=int, default=1, help='Number of threads to use for training')
    return parser.parse_args()

def preprocess_data(df):
    # Dropping redundant columns
    df = df.drop(['Result', 'Commentaries', 'Moves'], axis=1)

    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Adding new features
    df['white_time_min'] = df.apply(lambda row: time_to_minutes(row["White Clock"]), axis=1)
    df['black_time_min'] = df.apply(lambda row: time_to_minutes(row["Black Clock"]), axis=1)
    df['state_time'] = df.apply(lambda row: start_time(row["Time"]), axis=1)

    # Handling null values
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # Mapping target column to numeric
    unique_labels = df['Result-Winner'].unique()
    label_mapping = {label: index for index, label in enumerate(unique_labels)}
    df['Result-Winner'] = df['Result-Winner'].map(label_mapping)

    # Keeping only numeric columns for model training
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_columns]
    return df

def time_to_minutes(clock_str):
    try:
        hours, minutes = map(int, clock_str.split('.')[0].split(':'))
        return hours * 60 + minutes
    except:
        return None

def start_time(time_str):
    try:
        hours, minutes, seconds = map(int, time_str.split(':'))
        return hours
    except:
        return None

def train_model(X_train, y_train, num_threads):
    numeric_features = ['White Elo', 'Black Elo', 'White RD', 'Black RD', 'PlyCount', 'white_time_min', 'black_time_min', 'state_time']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numeric_transformer, numeric_features)
        ])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', SVC())])

    param_grid = [
        {'classifier': [SVC()], 'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']},
        {'classifier': [GaussianNB()]},
        {'classifier': [KNeighborsClassifier()], 'classifier__n_neighbors': [3, 5, 7]},
        {'classifier': [RandomForestClassifier()], 'classifier__n_estimators': [100, 200, 300], 'classifier__max_depth': [None, 5, 10]},
        {'classifier': [DecisionTreeClassifier()], 'classifier__max_depth': [None, 5, 10]},
        {'classifier': [LogisticRegressionCV()], 'classifier__solver': ['liblinear', 'lbfgs']},
        {'classifier': [MLPClassifier(random_state=42)], 'classifier__hidden_layer_sizes': [(10,), (10, 5)], 'classifier__activation': ['relu', 'tanh']}
    ]

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=num_threads)

    grid_search.fit(X_train, y_train)
    return grid_search

def evaluate_model(grid_search, X_test, y_test):
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    confusion = confusion_matrix(y_test, y_pred)

    print(f"Best Model: {best_model.named_steps['classifier'].__class__.__name__}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Test Accuracy: {accuracy}")
    print(f"Precision Score: {precision}")
    print(f"Recall Score: {recall}")
    print(f"F1 Score: {f1}")

def main(args):
    # Enable automatic logging
    mlflow.autolog()

    # Read the dataset
    df = pd.read_csv(args.data)
    df = df.iloc[:5000]  # Use a subset for quicker processing

    # Preprocess the data
    df = preprocess_data(df)

    y = df['Result-Winner']
    X = df.drop('Result-Winner', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    # Train the model
    grid_search = train_model(X_train, y_train, args.num_threads)

    # Evaluate the model
    evaluate_model(grid_search, X_test, y_test)
    
    # Saving the model with MLflow
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(grid_search.best_estimator_, "model")
        run_id = run.info.run_id

        

if __name__ == "__main__":
    args = parse_args()
    main(args)