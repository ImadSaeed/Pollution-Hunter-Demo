# src/inference_pipeline.py - Production inference for Pollution Hunter
# Works in both local and cloud environments (with src/ folder structure)

import numpy as np
import joblib
import json
from pathlib import Path

# Get the directory where THIS file is located (src/ folder)
# Then go up one level to reach project root
CURRENT_DIR = Path(__file__).parent      # src/ folder
PROJECT_ROOT = CURRENT_DIR.parent        # Root folder (where models/ is)
MODELS_DIR = PROJECT_ROOT / 'models'


class PollutionHunter:
    """
    Production-ready pollution prediction system.
    No disk I/O — all in-memory predictions.
    Supports both single and batch predictions.
    Works in local AND cloud environments.
    """
    
    def __init__(self, verbose=False):
        """
        Load all artifacts once at initialization.
        
        Args:
            verbose: If True, print loading status
        """
        self.verbose = verbose
        
        if self.verbose:
            print("="*50)
            print("🌍 Initializing Pollution Hunter")
            print("="*50)
        
        # Load model (relative path — works everywhere)
        model_path = MODELS_DIR / 'pollution_hunter_best_model.pkl'
        self.model = joblib.load(model_path)
        if self.verbose:
            print(f"✅ Model loaded: {type(self.model).__name__}")
        
        # Load feature names
        features_path = MODELS_DIR / 'feature_names.pkl'
        if features_path.exists():
            self.feature_names = joblib.load(features_path)
            if self.verbose:
                print(f"✅ Features: {len(self.feature_names)}")
        else:
            self.feature_names = None
            if self.verbose:
                print("⚠️ Feature names file not found")
        
        # Load city encoder
        encoder_path = MODELS_DIR / 'city_encoder.pkl'
        self.city_encoder = joblib.load(encoder_path)
        if self.verbose:
            print(f"✅ Cities: {list(self.city_encoder.classes_)}")
        
        # Load metadata (optional)
        metadata_path = MODELS_DIR / 'model_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            if self.verbose:
                print(f"✅ Model: {self.metadata.get('model_type', 'XGBoost')} (R² = {self.metadata.get('test_r2', 0.9827):.4f})")
        else:
            self.metadata = {'model_type': 'XGBoost', 'test_r2': 0.9827}
            if self.verbose:
                print("✅ Model: XGBoost (R² = 0.9827)")
        
        if self.verbose:
            print("="*50)
            print("✅ Pollution Hunter Ready!")
            print("="*50)
    
    def predict(self, city, temperature, humidity, wind_speed, pressure,
                hour, month, is_weekend,
                lag1, lag2, lag3, lag6, lag12, lag24,
                rolling_mean_3, rolling_mean_6, rolling_mean_12, rolling_std_6):
        """
        Predict PM2.5 for given inputs.
        
        Returns:
            dict: {'pm2.5': float, 'aqi_category': str, 'health_advice': str}
        """
        # Encode city
        try:
            city_encoded = self.city_encoder.transform([city])[0]
        except ValueError:
            available = list(self.city_encoder.classes_)
            raise ValueError(f"City '{city}' not found. Available: {available}")
        
        # Create feature array (order must match training)
        features = [
            temperature, humidity, wind_speed, pressure,
            hour, month, is_weekend,
            lag1, lag2, lag3, lag6, lag12, lag24,
            rolling_mean_3, rolling_mean_6, rolling_mean_12, rolling_std_6,
            city_encoded
        ]
        
        # Predict
        features_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features_array)[0]
        
        return {
            'pm2.5': round(prediction, 1),
            'aqi_category': self._get_aqi_category(prediction),
            'health_advice': self._get_health_advice(prediction)
        }
    
    def predict_batch(self, inputs_list):
        """
        Batch prediction from list of inputs (in-memory, no disk I/O).
        
        Args:
            inputs_list: List of dictionaries, each containing prediction parameters
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for inputs in inputs_list:
            result = self.predict(
                city=inputs['city'],
                temperature=inputs['temperature'],
                humidity=inputs['humidity'],
                wind_speed=inputs['wind_speed'],
                pressure=inputs['pressure'],
                hour=inputs['hour'],
                month=inputs['month'],
                is_weekend=inputs['is_weekend'],
                lag1=inputs['lag1'],
                lag2=inputs['lag2'],
                lag3=inputs['lag3'],
                lag6=inputs['lag6'],
                lag12=inputs['lag12'],
                lag24=inputs['lag24'],
                rolling_mean_3=inputs['rolling_mean_3'],
                rolling_mean_6=inputs['rolling_mean_6'],
                rolling_mean_12=inputs['rolling_mean_12'],
                rolling_std_6=inputs['rolling_std_6']
            )
            results.append(result)
        
        return results
    
    def _get_aqi_category(self, pm25):
        """Convert PM2.5 to AQI category"""
        if pm25 <= 15:
            return "Good"
        elif pm25 <= 35:
            return "Moderate"
        elif pm25 <= 55:
            return "Unhealthy for Sensitive"
        elif pm25 <= 150:
            return "Unhealthy"
        else:
            return "Hazardous"
    
    def _get_health_advice(self, pm25):
        """Get health advice based on PM2.5 level"""
        if pm25 <= 15:
            return "Air quality is good. Enjoy outdoor activities."
        elif pm25 <= 35:
            return "Air quality is acceptable. Sensitive individuals should limit prolonged outdoor exertion."
        elif pm25 <= 55:
            return "Members of sensitive groups may experience health effects. Limit outdoor activities."
        elif pm25 <= 150:
            return "Everyone may begin to experience health effects. Avoid prolonged outdoor activities."
        else:
            return "Health alert: everyone may experience serious health effects. Stay indoors."
    
    def get_available_cities(self):
        """Return list of available cities"""
        return list(self.city_encoder.classes_)
    
    def get_model_info(self):
        """Return model metadata"""
        return {
            'model_type': self.metadata.get('model_type', 'XGBoost'),
            'r2_score': self.metadata.get('test_r2', 0.9827),
            'features_count': len(self.feature_names) if self.feature_names else 18,
            'cities_count': len(self.city_encoder.classes_)
        }


# Optional: Quick test when run directly
if __name__ == "__main__":
    print("🧪 Testing Pollution Hunter...\n")
    
    # Initialize
    ph = PollutionHunter(verbose=True)
    
    # Test single prediction
    result = ph.predict(
        city='Islamabad',
        temperature=25.0, humidity=50.0, wind_speed=5.0, pressure=1015.0,
        hour=14, month=1, is_weekend=0,
        lag1=45.0, lag2=44.0, lag3=46.0, lag6=47.0, lag12=48.0, lag24=50.0,
        rolling_mean_3=45.0, rolling_mean_6=46.0, rolling_mean_12=47.0, rolling_std_6=2.0
    )
    
    print("\n📊 Single Prediction Test:")
    print(f"   PM2.5: {result['pm2.5']} μg/m³")
    print(f"   AQI: {result['aqi_category']}")
    print(f"   Advice: {result['health_advice']}")
    
    print("\n✅ Inference pipeline ready for import!")
    print("   Use: from src.inference_pipeline import PollutionHunter")