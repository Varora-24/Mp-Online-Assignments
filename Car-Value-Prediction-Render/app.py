from flask import Flask, request, jsonify, render_template_string
import pickle
import os

app = Flask(__name__)

# Load the trained ML model
MODEL_PATH = 'car_model.pkl'
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    model = None

@app.route('/', methods=['GET'])
def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Car Value Predictor API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; line-height: 1.6; }
            h1 { color: #2c3e50; }
            .endpoint { background: #f4f4f4; padding: 15px; border-radius: 5px; border-left: 5px solid #3498db; }
        </style>
    </head>
    <body>
        <h1>🚗 Car Value Predictor API (Render Deployment)</h1>
        <p>Welcome to the Car Value Prediction API. This API uses a Machine Learning model to predict the resale value of a car based on its year, mileage, and engine size.</p>
        
        <div class="endpoint">
            <h3>POST /predict</h3>
            <p>Send a JSON payload to get a price prediction.</p>
            <p><strong>Example Request:</strong></p>
            <code>
            {<br>
                &nbsp;&nbsp;"year": 2018,<br>
                &nbsp;&nbsp;"mileage": 45000,<br>
                &nbsp;&nbsp;"engine_size": 2.5<br>
            }
            </code>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded. Please run train.py first.'}), 500
        
    try:
        data = request.get_json()
        
        # Extract features
        year = float(data['year'])
        mileage = float(data['mileage'])
        engine_size = float(data['engine_size'])
        
        # Make prediction
        # The model expects a 2D array: [[year, mileage, engine_size]]
        prediction = model.predict([[year, mileage, engine_size]])
        
        predicted_price = round(prediction[0], 2)
        
        return jsonify({
            'success': True,
            'predicted_price_usd': predicted_price,
            'input_data': data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    # Use the port assigned by Render, or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
