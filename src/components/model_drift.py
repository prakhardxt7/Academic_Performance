from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import os
import logging
import mlflow
import pandas as pd
from datetime import datetime
from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab
from sklearn.metrics import r2_score
from src.utils import load_object
from src.logger import logging
import time
class ModelDrift:
    def __init__(self):
        self.model_file_path = "artifacts/model.pkl"  # Fixed model path
        self.drift_threshold = 0.4
        self.drift_names = {
            "data_drift": "Overall Data Drift",
            "feature_drift": "Feature Drift",
            "target_drift": "Target Drift",
        }

    def check_drift(self, new_data):
        try:
            # Assuming 'new_data' is a DataFrame with new data for drift analysis
            column_mapping = ColumnMapping()
            column_mapping.target = "math_score"  # Define the target column
            dashboard = Dashboard(tabs=[DataDriftTab(), CatTargetDriftTab()])

            # Generate the drift report
            drift_report = dashboard.calculate(new_data, column_mapping=column_mapping)
            drift_metrics = drift_report.to_dict()  # Extract drift metrics

            # Log drift metrics to console
            logging.info(f"Evidently Drift Metrics: {drift_metrics}")

            # Log drift metrics to MLflow
            mlflow.log_metric("data_drift_score", drift_metrics.get("data_drift", {}).get("drift_score", 0))
            mlflow.log_metric("feature_drift_score", drift_metrics.get("feature_drift", {}).get("drift_score", 0))
            mlflow.log_metric("target_drift_score", drift_metrics.get("target_drift", {}).get("drift_score", 0))

            # Log p-values and other relevant drift metrics
            mlflow.log_metric("data_drift_p_value", drift_metrics.get("data_drift", {}).get("p_value", 0))
            mlflow.log_metric("feature_drift_p_value", drift_metrics.get("feature_drift", {}).get("p_value", 0))
            mlflow.log_metric("target_drift_p_value", drift_metrics.get("target_drift", {}).get("p_value", 0))

            # Check if drift threshold is exceeded
            if drift_metrics.get("data_drift", {}).get("drift_score", 0) > self.drift_threshold:
                logging.info(f"Data Drift detected, score: {drift_metrics['data_drift']['drift_score']}")
                return True
            else:
                logging.info(f"No significant drift detected, score: {drift_metrics['data_drift']['drift_score']}")
                return False

        except Exception as e:
            logging.error(f"Error occurred while checking drift: {str(e)}", exc_info=True)
            raise e

# Main pipeline
def train_pipeline():
    try:
        logging.info("Starting the training pipeline.")

        # Step 1: Data Ingestion (assuming data_ingestion is already defined)
        logging.info("Step 1: Starting Data Ingestion...")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Step 1 Completed: Data ingestion completed. Train data path: {train_data_path}")

        # Step 2: Data Transformation (assuming data_transformation is already defined)
        logging.info("Step 2: Starting Data Transformation...")
        start_time = time.time()
        data_transformation = DataTransformation()
        train_array, _, preprocessor = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info(f"Step 2 Completed: Data transformation completed in {time.time() - start_time:.2f} seconds.")

        # Save preprocessor to ensure consistency
        preprocessor_path = 'artifacts/preprocessor.pkl'
        preprocessor = load_object(preprocessor_path)  # Ensure it's properly loaded
        if not hasattr(preprocessor, "transform"):
            raise TypeError("Preprocessor object is invalid. Check the saved artifact.")

        # Step 3: Model Training (using existing model)
        logging.info("Step 3: Loading existing model...")
        model = load_object('artifacts/model.pkl')  # Load the pre-trained model
        logging.info("Step 3 Completed: Existing model loaded successfully.")

        # Step 4: Drift Check on Synthetic Data
        logging.info("Step 4: Checking for Drift on new synthetic data...")
        drift_checker = ModelDrift()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        drift_file_path = os.path.join(current_dir, "data", "new_data.csv")

        new_data = pd.read_csv(drift_file_path)

        # Step 5: Check drift and compare R² score
        #if drift_checker.check_drift(new_data):  # If drift detected
        #    logging.info("Drift detected. Using existing model for prediction...")
        #else:
         #   logging.info("No significant drift detected. Using existing model for prediction.")

        # Compare R² score on synthetic data and log it in MLflow
        X_synthetic = new_data.drop(columns=["math_score"])
        y_synthetic = new_data["math_score"]
        X_synthetic_transformed = preprocessor.transform(X_synthetic)

        # Predictions using the existing model
        predictions = model.predict(X_synthetic_transformed)
        r2_score_synthetic = r2_score(y_synthetic, predictions)
        logging.info(f"R² Score on synthetic data: {r2_score_synthetic:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("r2_score_synthetic", r2_score_synthetic)

        logging.info("Training pipeline completed successfully.")

        return "model.pkl", r2_score_synthetic

    except Exception as e:
        logging.error("Error occurred in the training pipeline.", exc_info=True)
        raise e

if __name__ == "__main__":
    logging.info("Setting MLflow Tracking URI to port 5002")
    mlflow.set_tracking_uri("http://localhost:5002")  # Set MLflow Tracking URI to port 5002

    result = train_pipeline()
    print(f"Pipeline completed.")
    print(f"Best Model: {result[0]}")
    print(f"R2 Score on synthetic data: {result[1]:.4f}")


