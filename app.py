from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

FEATURE_NAMES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

DEFAULT_VALUES = {
    'mean radius': 13.54, 'mean texture': 14.36, 'mean perimeter': 87.46, 'mean area': 566.3,
    'mean smoothness': 0.09779, 'mean compactness': 0.08129, 'mean concavity': 0.06664,
    'mean concave points': 0.04781, 'mean symmetry': 0.1885, 'mean fractal dimension': 0.05766,
    'radius error': 0.2699, 'texture error': 0.7886, 'perimeter error': 2.058, 'area error': 23.56,
    'smoothness error': 0.008462, 'compactness error': 0.0146, 'concavity error': 0.02387,
    'concave points error': 0.01315, 'symmetry error': 0.0198, 'fractal dimension error': 0.0023,
    'worst radius': 15.11, 'worst texture': 19.26, 'worst perimeter': 99.7, 'worst area': 711.2,
    'worst smoothness': 0.144, 'worst compactness': 0.1773, 'worst concavity': 0.239,
    'worst concave points': 0.1288, 'worst symmetry': 0.2977, 'worst fractal dimension': 0.07259
}

class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model_and_scaler()

    def load_model_and_scaler(self):
        try:
            self.model = joblib.load('models/model.pkl')
            self.scaler = joblib.load('models/log_scaler.pkl')
            app.logger.info("Model and scaler loaded successfully")
        except Exception as e:
            app.logger.error(f"Failed to load model or scaler: {e}")
            self.model = None
            self.scaler = None

    def is_loaded(self):
        return self.model is not None and self.scaler is not None

    def predict(self, features):
        features_array = np.array(features).reshape(1, -1)
        scaled_features = self.scaler.transform(features_array)
        prediction = self.model.predict(scaled_features)[0]
        probability = self.model.predict_proba(scaled_features)[0][1]
        return prediction, probability

model_manager = ModelManager()

@app.route('/')
def home():
    return render_template('index.html', feature_names=FEATURE_NAMES, default_values=DEFAULT_VALUES)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_manager.is_loaded():
        app.logger.error("Model or scaler not loaded properly")
        return jsonify({'error': 'Model or scaler not loaded properly'}), 500

    try:
        data = request.get_json()
        features = data.get('features', [])

        if len(features) != 30:
            app.logger.warning("Invalid number of features provided")
            return jsonify({'error': 'Exactly 30 features are required'}), 400

        prediction, probability = model_manager.predict(features)

        return jsonify({
            'prediction': int(prediction),
            'probability': round(float(probability), 4),
            'diagnosis': 'Malignant' if prediction == 1 else 'Benign'
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# Run app
if __name__ == '__main__':
    app.run(debug=True)