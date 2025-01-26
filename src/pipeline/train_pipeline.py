from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.utils import load_object
from sklearn.metrics import r2_score
import pandas as pd
import time

def train_pipeline():
    try:
        logging.info("Starting the training pipeline.")

        # Step 1: Data Ingestion
        logging.info("Step 1: Starting Data Ingestion...")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Step 1 Completed: Data ingestion completed. Train data path: {train_data_path}, Test data path: {test_data_path}")

        # Step 2: Data Transformation
        logging.info("Step 2: Starting Data Transformation...")
        start_time = time.time()
        data_transformation = DataTransformation()
        train_array, test_array, preprocessor = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info(f"Step 2 Completed: Data transformation completed in {time.time() - start_time:.2f} seconds.")

        # Save preprocessor to ensure consistency
        preprocessor_path = 'artifacts/preprocessor.pkl'
        preprocessor = load_object(preprocessor_path)  # Ensure it's properly loaded
        if not hasattr(preprocessor, "transform"):
            raise TypeError("Preprocessor object is invalid. Check the saved artifact.")

        # Step 3: Model Training
        logging.info("Step 3: Starting Model Training...")
        start_time = time.time()
        model_trainer = ModelTrainer()
        best_model_name, r2_square_train = model_trainer.initiate_model_trainer(train_array, test_array)
        logging.info(f"Step 3 Completed: Model training completed in {time.time() - start_time:.2f} seconds. Best model: {best_model_name}, R2 Score on Train/Test: {r2_square_train}")

        # Step 4: Evaluate on artifacts/test.csv
        logging.info("Step 4: Evaluating Best Model on artifacts/test.csv...")
        test_csv_path = 'artifacts/test.csv'
        test_csv_data = pd.read_csv(test_csv_path)
        target_column = "math_score"

        if target_column in test_csv_data.columns:
            X_test = test_csv_data.drop(columns=[target_column])
            y_test = test_csv_data[target_column]
        else:
            raise ValueError(f"Target column '{target_column}' not found in test.csv.")

        # Transforming the test data using the preprocessor
        X_test_transformed = preprocessor.transform(X_test)

        # Loading the trained model (always use model.pkl)
        best_model_path = 'artifacts/model.pkl'  # Fixed to model.pkl
        best_model = load_object(best_model_path)

        # Making predictions and calculating R² score
        predictions = best_model.predict(X_test_transformed)
        r2_score_test_csv = r2_score(y_test, predictions)
        logging.info(f"Step 4 Completed: R2 Score on artifacts/test.csv: {r2_score_test_csv:.4f}")

        logging.info("Training pipeline completed successfully.")

        return best_model_name, r2_square_train, r2_score_test_csv

    except Exception as e:
        logging.error("Error occurred in the training pipeline.", exc_info=True)
        raise e

if __name__ == "__main__":
    result = train_pipeline()
    print(f"Pipeline completed.")
    print(f"Best Model: {result[0]}")
    print(f"R2 Score on Train/Test Split: {result[1]:.4f}")
    print(f"R2 Score on artifacts/test.csv: {result[2]:.4f}")
