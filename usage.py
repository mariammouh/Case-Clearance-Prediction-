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

        processed_data = new_data_df[self.feature_names].copy()
        

        if self.scaler:
            processed_data = self.scaler.transform(processed_data)
        
        return processed_data
    
    def predict(self, new_cases):
        """Make predictions on new crime cases"""

        processed_data = self.preprocess_new_data(new_cases)
        
    
        predictions = self.model.predict(processed_data)
        prediction_probas = self.model.predict_proba(processed_data)
        

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


if __name__ == "__main__":
 
    predictor = CrimePredictor()

    print("Model expects these features:")
    print(predictor.feature_names)




    new_cases_example = pd.DataFrame({

    **{feature: [0] for feature in predictor.feature_names},  
    
   
    'TIME OCC': [1430],                    
    'AREA': [3],                           
    'Rpt Dist No': [217],                  
    'Part 1-2': [1],                       
    'Crm Cd': [624],                       
    'Vict Age': [35],                      
    'time_to_report': [2.5],               
    'crime_count': [1],                   
    'location_cluster': [25],             
    
   
    'Vict Sex_M': [1],                
    'Vict Sex_F': [0], 'Vict Sex_H': [0], 'Vict Sex_X': [0], 'Vict Sex_-': [0],  
 
    'Vict Descent_B': [1],            
    'Vict Descent_-': [0], 'Vict Descent_A': [0], 'Vict Descent_C': [0], 
    'Vict Descent_D': [0], 'Vict Descent_F': [0], 'Vict Descent_G': [0],
    'Vict Descent_H': [0], 'Vict Descent_I': [0], 'Vict Descent_J': [0],
    'Vict Descent_K': [0], 'Vict Descent_L': [0], 'Vict Descent_O': [0],
    'Vict Descent_P': [0], 'Vict Descent_S': [0], 'Vict Descent_U': [0],
    'Vict Descent_V': [0], 'Vict Descent_W': [0], 'Vict Descent_X': [0],
    'Vict Descent_Z': [0],
    
   
    'Premis Cd_501.0': [1],              
    
    'mo_embedding_0': [0.1], 'mo_embedding_1': [0.2], 'mo_embedding_2': [0.3],
    'mo_embedding_3': [0.4], 'mo_embedding_4': [0.5], 'mo_embedding_5': [0.6],

})

print(f"Created example with {len(new_cases_example.columns)} features")
print("Example shape:", new_cases_example.shape)


missing_features = set(predictor.feature_names) - set(new_cases_example.columns)
extra_features = set(new_cases_example.columns) - set(predictor.feature_names)

print(f"Missing features: {len(missing_features)}")
print(f"Extra features: {len(extra_features)}")

if len(missing_features) == 0:
    print("✅ All required features are present!")
    
   
    predictions = predictor.predict(new_cases_example)
    
    
    print("\nPrediction Results:")
    for result in predictions:
        print(f"\nCase {result['case_id']}:")
        print(f"  Predicted: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print("  Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"    {cls}: {prob:.2%}")
else:
    print("❌ Missing features:", missing_features)
    
 
    predictions = predictor.predict(new_cases_example)
    

    print("Prediction Results:")
    for result in predictions:
        print(f"\nCase {result['case_id']}:")
        print(f"  Predicted: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print("  Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"    {cls}: {prob:.2%}")