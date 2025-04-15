import pandas as pd
import joblib
import os
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    IsolationForest
)
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

MODEL_DIR = 'prediction_app/ml_models'

def train_from_dataframes(self, X, y):
    """Train model directly from pandas DataFrames"""
    try:
        required_inputs = ['COD', 'pH', 'TSS', 'TDS', 'Conductivity']
        required_outputs = ['Effluent_COD', 'Effluent_pH', 'Effluent_TSS',
                          'Effluent_TDS', 'Effluent_Conductivity']

        missing_in = [col for col in required_inputs if col not in X.columns]
        missing_out = [col for col in required_outputs if col not in y.columns]

        if missing_in:
            raise ValueError(f"Missing in influent data: {missing_in}")
        if missing_out:
            raise ValueError(f"Missing in effluent data: {missing_out}")

        X_clean, y_clean = self._preprocess_data(X[required_inputs], y[required_outputs])
        self._train_models(X_clean, y_clean)
        self.is_trained = True
        return True
    except Exception as e:
        raise RuntimeError(f"Training failed: {str(e)}")
class EffluentPredictor:
    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.models = {
            'RF': RandomForestRegressor(n_estimators=100, random_state=42),
            'ET': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'XGB': XGBRegressor(n_estimators=100, random_state=42)
        }
        self.preprocessor = {
            'imputer': SimpleImputer(strategy='median'),
            'scaler': RobustScaler(),
            'outlier': IsolationForest(contamination=0.1, random_state=42)
        }
        self.is_trained = False

    def train_from_excel(self, influent_file, effluent_file):
        """Train model from uploaded Excel files"""
        try:
            # Load data
            X = pd.read_excel(influent_file)
            y = pd.read_excel(effluent_file)

            # Validate columns
            required_inputs = ['COD', 'pH', 'TSS', 'TDS', 'Conductivity']
            required_outputs = ['Effluent_COD', 'Effluent_pH', 'Effluent_TSS',
                                'Effluent_TDS', 'Effluent_Conductivity']

            missing_in = [col for col in required_inputs if col not in X.columns]
            missing_out = [col for col in required_outputs if col not in y.columns]

            if missing_in:
                raise ValueError(f"Missing in influent data: {missing_in}")
            if missing_out:
                raise ValueError(f"Missing in effluent data: {missing_out}")

            # Preprocess and train
            X_clean, y_clean = self._preprocess_data(X[required_inputs], y[required_outputs])
            self._train_models(X_clean, y_clean)
            self.is_trained = True
            return True
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")

    def predict(self, input_data):
        """Predict effluent from input parameters"""
        if not self.is_trained and not self._check_models_exist():
            raise ValueError("Model not trained. Please train first.")

        try:
            # Convert to DataFrame
            X = pd.DataFrame([input_data.values()], columns=input_data.keys())

            # Load models
            model, preprocessor = self._load_models()

            # Preprocess and predict
            X_imp = preprocessor['imputer'].transform(X)
            X_scaled = preprocessor['scaler'].transform(X_imp)
            preds = model.predict(X_scaled)

            return pd.DataFrame(
                preds,
                columns=['Effluent_' + col for col in input_data.keys()]
            ).iloc[0].to_dict()
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    # ===== Internal Methods =====
    def _preprocess_data(self, X, y):
        """Clean and validate data"""
        X_imp = pd.DataFrame(
            self.preprocessor['imputer'].fit_transform(X),
            columns=X.columns
        )
        outliers = self.preprocessor['outlier'].fit_predict(X_imp)
        return X_imp[outliers == 1], y.iloc[outliers == 1]

    def _train_models(self, X, y):
        """Train with train-test split"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_scaled = self.preprocessor['scaler'].fit_transform(X_train)

        best_score = -1
        best_model = None

        for name, model in self.models.items():
            model.fit(X_scaled, y_train)
            score = model.score(
                self.preprocessor['scaler'].transform(X_test),
                y_test
            )
            if score > best_score:
                best_score = score
                best_model = model

        # Final training on full data
        best_model.fit(
            self.preprocessor['scaler'].fit_transform(X),
            y
        )
        self._save_models(best_model)

    def _save_models(self, model):
        """Persist models"""
        joblib.dump(model, f'{MODEL_DIR}/model.joblib')
        joblib.dump(self.preprocessor, f'{MODEL_DIR}/preprocessor.joblib')

    def _load_models(self):
        """Load trained models"""
        return (
            joblib.load(f'{MODEL_DIR}/model.joblib'),
            joblib.load(f'{MODEL_DIR}/preprocessor.joblib')
        )

    def _check_models_exist(self):
        return all([
            os.path.exists(f'{MODEL_DIR}/model.joblib'),
            os.path.exists(f'{MODEL_DIR}/preprocessor.joblib')
        ])