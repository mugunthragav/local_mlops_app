import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from models.models_config import models

# Set the MLflow tracking URI to a writable local path
mlflow.set_tracking_uri(os.path.join(os.getcwd(), "mlruns"))


def load_data():
    # Use relative path to access the CSV file in the GitHub repository workspace
    local_file = os.path.join(os.getenv("GITHUB_WORKSPACE", "."), "Housing.csv")

    if not os.path.exists(local_file):
        print(f"Data file {local_file} does not exist.")
        raise FileNotFoundError(f"Data file {local_file} not found.")

    data = pd.read_csv(local_file)

    # Convert integer columns to float if they may have missing values
    for col in data.select_dtypes(include=['int']).columns:
        data[col] = data[col].astype(float)

    return data


def train_all_models(data):
    experiment_name = "House_Pricing_Model_local"
    mlflow.set_experiment(experiment_name)

    # Preprocessing: Encode all categorical features
    categorical_columns = ['mainroad', 'guestroom', 'furnishingstatus', 'airconditioning', 'basement', 'hotwaterheating', 'parking', 'prefarea']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    X = data.drop('price', axis=1)
    y = data['price']

    model_performance = []

    for model_name, model_config in models.items():
        print(f"Training model: {model_name}")

        model_class = model_config["model"]
        model_params = model_config["parameters"]

        test_size = model_params.pop('test_size', 0.2)
        random_state = model_params.pop('random_state', 42)

        with mlflow.start_run(run_name=model_name):
            mlflow.autolog()

            model = model_class(**model_params)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            rmse = mean_squared_error(y_test, predictions, squared=False)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            model_performance.append({
                "model_name": model_name,
                "model": model,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "model_class_name": model.__class__.__name__
            })

    best_model = min(model_performance, key=lambda x: x['rmse'])
    print(f"Best Model: {best_model['model_name']} with RMSE: {best_model['rmse']}")

    # Log the best model
    example_input = X_test.head(1)  # Add example input for signature
    mlflow.sklearn.log_model(best_model['model'], artifact_path="$GITHUB_WORKSPACE/mlruns", input_example=example_input)

    best_model_uri = f"runs:/{mlflow.active_run().info.run_id}/mlruns/best_model"

    # Save best model details to JSON
    with open("best_model_info.json", "w") as f:
        json.dump({
            "model_name": best_model['model_class_name'],
            "rmse": best_model['rmse'],
            "mae": best_model['mae'],
            "r2": best_model['r2'],
            "model_uri": best_model_uri
        }, f)

    return best_model['model_class_name'], best_model['model']


if __name__ == "__main__":
    data = load_data()
    best_model_name, best_model = train_all_models(data)
