import mlflow
import mlflow.sklearn
import json  # To load best model info
from model_training import train_all_models  # Import the training function
from mlflow.tracking import MlflowClient
from model_training import load_data


def promote_and_register_model(best_model_name, best_model, rmse, mae, r2):
    """
    Logs the best model and its metrics, registers it to MLflow, and promotes it to Production.
    """
    # Print model performance
    print(f"Promoting and registering model: {best_model_name}")
    print(f"Model performance - RMSE: {rmse}, MAE: {mae}, R²: {r2}")

    with mlflow.start_run():  # Start an MLflow run
        # Log model metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log the best model in the MLflow registry and register it
        mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model_name)
        print(f"Model {best_model_name} successfully registered!")

    # Create an MLflow Client to interact with the Model Registry
    client = MlflowClient()

    # Get the latest model version (after registering the model)
    model_versions = client.get_latest_versions(best_model_name, stages=["None"])  # Get the latest unassigned versions
    model_version = model_versions[0].version if model_versions else None

    if model_version is None:
        print(f"Error: No version found for model {best_model_name}.")
        return

    # Promote the model to the "Production" stage
    client.transition_model_version_stage(
        name=best_model_name,
        version=model_version,
        stage="Production",
        archive_existing_versions=True  # Archives previously promoted models
    )

    print(f"Model {best_model_name} version {model_version} promoted to Production!")


if __name__ == "__main__":
    # Assuming that you have the training function already defined in 'train_all_models'
    data = load_data()  # Load your data

    # Train all models and get the best model information
    best_model_name, best_model = train_all_models(data)

    # Load best model metrics (replace this with actual metrics from your training function)
    # Assuming you already calculate RMSE, MAE, R² during the training process.
    best_model_metrics = {
        "rmse": 0.12,  # Example RMSE value
        "mae": 0.08,   # Example MAE value
        "r2": 0.95     # Example R² value
    }

    # Call the promote_and_register_model function
    promote_and_register_model(
        best_model_name=best_model_name,
        best_model=best_model,
        rmse=best_model_metrics["rmse"],
        mae=best_model_metrics["mae"],
        r2=best_model_metrics["r2"]
    )
