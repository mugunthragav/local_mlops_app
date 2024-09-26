import os
import argparse
import mlflow
import mlflow.sklearn
import json  # To save model information
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from models.models_config import models

# Set the MLflow tracking URI to a local path
mlflow.set_tracking_uri("file:///C:/Users/Admin/PycharmProjects/mlflow/local_mlops/mlruns")


def load_data():
    local_file = "Housing.csv"  # Local path to your data file
    if not os.path.exists(local_file):
        print(f"Data file {local_file} does not exist.")
        raise FileNotFoundError(f"Data file {local_file} not found.")
    return pd.read_csv(local_file)


def train_all_models(data):
    experiment_name = "House_Pricing_Model_local"
    mlflow.set_experiment(experiment_name)

    X = data.drop('price', axis=1)  # Separate features
    y = data['price']  # Target variable

    print("Checking for missing values...")
    print(data.isnull().sum())

    # Convert all columns to numeric
    X = X.apply(pd.to_numeric, errors='coerce')

    model_performance = []  # Store model performance

    for model_name, model_config in models.items():
        print(f"Training model: {model_name}")

        model_class = model_config["model"]  # Get model class
        model_params = model_config["parameters"]

        test_size = model_params.pop('test_size', 0.2)
        random_state = model_params.pop('random_state', 42)

        with mlflow.start_run(run_name=model_name):
            mlflow.autolog()  # Ensure autologging is set before model training

            model = model_class(**model_params)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            print(f"Number of predictions: {len(predictions)}")

            rmse = mean_squared_error(y_test, predictions, squared=False)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            print(f"Model {model_name} trained successfully with RMSE: {rmse}, MAE: {mae}, R²: {r2}")

            # Store performance details
            model_performance.append({
                "model_name": model_name,
                "model": model,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "model_class_name": model.__class__.__name__  # Get the model class name
            })

    # Find the best model based on RMSE
    best_model = min(model_performance, key=lambda x: x['rmse'])
    print(f"Best Model: {best_model['model_name']} with RMSE: {best_model['rmse']}")

    # Log the best model in MLflow and save the model URI
    mlflow.sklearn.log_model(best_model['model'], artifact_path="best_model")
    best_model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"

    # End the previous run
    mlflow.end_run()

    # Start a new run for the final best model registration
    with mlflow.start_run(run_name=f"{best_model['model_name']}_final"):
        mlflow.sklearn.log_model(best_model['model'], "model", registered_model_name=best_model['model_name'])

    # Save best model details to JSON, including the model URI
    with open("best_model_info.json", "w") as f:
        json.dump({
            "model_name": best_model['model_class_name'],  # Use class name for model
            "rmse": best_model['rmse'],
            "mae": best_model['mae'],
            "r2": best_model['r2'],
            "model_uri": best_model_uri  # Save the URI of the best model
        }, f)

    # Return the best model class name and instance
    return best_model['model_class_name'], best_model['model']


# Main function
if __name__ == "__main__":
    data = load_data()  # Load data from local file
    best_model_name, best_model = train_all_models(data)

