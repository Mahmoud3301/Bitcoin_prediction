from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib
import datetime as dt
import json
import os

app = Flask(__name__)

# PyTorch Model Definitions
class BitcoinLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(BitcoinLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.linear(out)
        return out

class BitcoinGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(BitcoinGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.linear(out)
        return out

# Global variables
lstm_model = None
gru_model = None
scaler = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models():
    """Load pre-trained models and scaler"""
    global lstm_model, gru_model, scaler
    
    try:
        # Check if models directory exists
        if not os.path.exists('models'):
            print("Models directory not found. Running in demo mode.")
            return False
            
        # Load LSTM model
        lstm_path = 'C:\\Mahmoud_Saeed\\My_projects\\deep\\models\\1d_bitcoin_lstm_model.pth'
        if os.path.exists(lstm_path):
            lstm_model = BitcoinLSTM()
            lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
            lstm_model.eval()
            print("LSTM model loaded successfully")
        else:
            print(f"LSTM model not found at {lstm_path}")
            return False
        
        # Load GRU model
        gru_path = r'C:\\Mahmoud_Saeed\\My_projects\\deep\\models\\1d_bitcoin_gru_model.pth'
        if os.path.exists(gru_path):
            gru_model = BitcoinGRU()
            gru_model.load_state_dict(torch.load(gru_path, map_location=device))
            gru_model.eval()
            print("GRU model loaded successfully")
        else:
            print(f"GRU model not found at {gru_path}")
            return False
        
        # Load scaler
        scaler_path = 'C:\\Mahmoud_Saeed\\My_projects\\deep\\models\\1d_bitcoin_scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("Scaler loaded successfully")
        else:
            print(f"Scaler not found at {scaler_path}")
            return False
        
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Load models when app starts
models_loaded = load_models()
if not models_loaded:
    print("Running in demo mode with mock predictions")

def get_sample_data():
    """Get sample Bitcoin data for prediction"""
    # Generate realistic sample data
    base_price = 45000
    volatility = 2000
    days = 60
    
    # Generate sample data with some trend and noise
    np.random.seed(42)
    sample_data = []
    price = base_price
    
    for i in range(days):
        # Add some trend and random noise
        change = np.random.normal(50, volatility * 0.1)
        price += change
        price = max(price, 10000)
        sample_data.append(price)
    
    return np.array(sample_data[-60:])

def predict_future(model, scaler, last_sequence, future_days=30, time_step=60):
    """Predict future prices"""
    if model is None or scaler is None:
        # Return mock predictions if models aren't loaded
        return generate_mock_predictions(last_sequence, future_days)
    
    model.eval()
    
    # Ensure we have the right sequence length
    if len(last_sequence) > time_step:
        last_sequence = last_sequence[-time_step:]
    elif len(last_sequence) < time_step:
        padding = np.full(time_step - len(last_sequence), last_sequence[0])
        last_sequence = np.concatenate([padding, last_sequence])
    
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    
    predictions = []
    current_sequence = last_sequence_scaled.copy()
    
    with torch.no_grad():
        for _ in range(future_days):
            input_data = torch.FloatTensor(current_sequence).view(1, time_step, 1)
            prediction = model(input_data)
            pred_value = prediction.numpy()[0, 0]
            current_sequence = np.append(current_sequence[1:], pred_value)
            predictions.append(pred_value)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

def generate_mock_predictions(last_sequence, future_days):
    """Generate realistic mock predictions"""
    if len(last_sequence) == 0:
        last_price = 45000
    else:
        last_price = last_sequence[-1]
    
    predictions = []
    current_price = last_price
    
    for i in range(future_days):
        change_percent = np.random.normal(0.002, 0.02)
        current_price = current_price * (1 + change_percent)
        current_price = max(current_price, 10000)
        predictions.append(current_price)
    
    return np.array(predictions)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data received'})
            
        model_type = data.get('model_type', 'LSTM')
        prediction_days = int(data.get('prediction_days', 30))
        
        print(f"Prediction request - Model: {model_type}, Days: {prediction_days}")
        
        # Validate inputs
        if model_type.upper() not in ['LSTM', 'GRU']:
            return jsonify({'success': False, 'error': 'Invalid model type'})
        
        if prediction_days not in [7, 15, 30, 60, 90]:
            return jsonify({'success': False, 'error': 'Invalid prediction days'})
        
        # Get sample data
        sample_sequence = get_sample_data()
        current_price = sample_sequence[-1] if len(sample_sequence) > 0 else 45000
        
        # Select model
        if model_type.upper() == 'LSTM':
            model = lstm_model
        else:
            model = gru_model
        
        # Make prediction
        predictions = predict_future(model, scaler, sample_sequence, prediction_days)
        
        # Generate dates
        last_date = dt.datetime.now()
        dates = [(last_date + dt.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, prediction_days + 1)]
        
        # Create response
        result = {
            'success': True,
            'model_used': model_type,
            'prediction_days': prediction_days,
            'current_price': round(float(current_price), 2),
            'predictions': [
                {
                    'date': date, 
                    'price': round(float(price), 2),
                    'change_percent': round(((price - current_price) / current_price * 100), 2)
                } 
                for date, price in zip(dates, predictions)
            ],
            'summary': {
                'min_price': round(float(np.min(predictions)), 2),
                'max_price': round(float(np.max(predictions)), 2),
                'avg_price': round(float(np.mean(predictions)), 2),
                'first_price': round(float(predictions[0]), 2),
                'last_price': round(float(predictions[-1]), 2),
                'overall_change': round(((predictions[-1] - current_price) / current_price * 100), 2)
            }
        }
        
        print(f"Prediction successful: {len(predictions)} days")
        return jsonify(result)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/current_price', methods=['GET'])
def current_price():
    """API endpoint for current Bitcoin price"""
    try:
        # Generate realistic data
        sample_data = get_sample_data()
        current_price = sample_data[-1] if len(sample_data) > 0 else 45280.50
        
        # Mock market data
        change_24h = np.random.uniform(-3, 5)
        market_cap = current_price * 19400000
        
        return jsonify({
            'success': True,
            'price': round(current_price, 2),
            'change_24h': round(change_24h, 2),
            'market_cap': f"${market_cap:,.0f}",
            'timestamp': dt.datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    print("=" * 60)
    print("Bitcoin Predictor 2025 - Starting Server...")
    print(f"Models loaded: {models_loaded}")
    print(f"Device: {device}")
    print("Server running on: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)