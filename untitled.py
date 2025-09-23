import joblib
import pandas as pd
import numpy as np

class CrimePredictor:
    def __init__(self, model_path='models/crime_prediction_model.pkl'):
        """Load the trained model and all necessary artifacts"""
        artifacts = joblib.load(model_path)
        self.model = artifacts['model']
        self.feature_names = artifacts['feature_names']
        self.class_names = artifacts['class_names']
        self.scaler = artifacts.get('scaler', None)
        print("Model loaded successfully!")
    
    def preprocess_new_data(self, new_data_df):
        """Preprocess new data to match training format"""
        # Ensure correct column order and features
        processed_data = new_data_df[self.feature_names].copy()
        
        # Apply same scaling used during training
        if self.scaler:
            processed_data = self.scaler.transform(processed_data)
        
        return processed_data
    
    def predict(self, new_cases):
        """Make predictions on new crime cases"""
        # Preprocess the new data
        processed_data = self.preprocess_new_data(new_cases)
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        prediction_probas = self.model.predict_proba(processed_data)
        
        # Convert to readable results
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, prediction_probas)):
            result = {
                'case_id': i,
                'predicted_class': self.class_names[pred],
                'confidence': float(np.max(proba)),
                'probabilities': {
                    self.class_names[j]: float(prob) 
                    for j, prob in enumerate(proba)
                }
            }
            results.append(result)
        
        return results
    
    def predict_single_case(self, case_features):
        """Predict for a single case (convenience method)"""
        return self.predict(pd.DataFrame([case_features]))[0]

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = CrimePredictor()
    
    # Example new case data (should match your feature engineering)
    new_cases_example = pd.DataFrame({
        'TIME OCC': [1430],
        'AREA': [3],
        'Rpt Dist No': [217],
        'Part 1-2': [1],
        'Crm Cd': [624],
        'Vict Age': [35],
        'Premis Cd_501.0': [1],
        'time_to_report_hours': [2.5],
        'crime_count': [1],
        'location_cluster': [25],
        'mo_count': [2],
        'mo_present_1822': [1],
        'mo_present_0344': [1]
        # ... include all your features in the exact same order
    })
    
    # Make predictions
    predictions = predictor.predict(new_cases_example)
    
    # Display results
    print("Prediction Results:")
    for result in predictions:
        print(f"\nCase {result['case_id']}:")
        print(f"  Predicted: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print("  Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"    {cls}: {prob:.2%}")