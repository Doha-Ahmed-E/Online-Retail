from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and scaler
MODEL_PATH = 'models/kmeans_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
SEGMENTS_PATH = 'data/cluster_segments.csv'

# Global variables to store loaded models
model = None
scaler = None
segment_info = None

def load_models():
    """Load the trained model, scaler, and segment information"""
    global model, scaler, segment_info
    
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        segment_info = pd.read_csv(SEGMENTS_PATH, index_col='Cluster')
        print("✓ Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Load models on startup
load_models()

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

@app.route('/api/segments', methods=['GET'])
def get_segments():
    """Get information about all customer segments"""
    try:
        if segment_info is None:
            return jsonify({'error': 'Segment information not loaded'}), 500
        
        segments = []
        for idx in segment_info.index:
            segments.append({
                'cluster_id': int(idx),
                'segment_name': segment_info.loc[idx, 'Segment'],
                'avg_recency': round(segment_info.loc[idx, 'Recency'], 2),
                'avg_frequency': round(segment_info.loc[idx, 'Frequency'], 2),
                'avg_monetary': round(segment_info.loc[idx, 'Monetary'], 2)
            })
        
        return jsonify({
            'success': True,
            'segments': segments
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_segment():
    """
    Predict customer segment based on RFM values
    Expected JSON input:
    {
        "recency": <days since last purchase>,
        "frequency": <number of purchases>,
        "monetary": <total spending>
    }
    """
    try:
        # Check if models are loaded
        if model is None or scaler is None:
            return jsonify({
                'success': False,
                'error': 'Models not loaded'
            }), 500
        
        # Get input data
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        required_fields = ['recency', 'frequency', 'monetary']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Extract features
        recency = float(data['recency'])
        frequency = float(data['frequency'])
        monetary = float(data['monetary'])
        
        # Validate values
        if recency < 0 or frequency < 0 or monetary < 0:
            return jsonify({
                'success': False,
                'error': 'All values must be non-negative'
            }), 400
        
        # Create feature array
        features = np.array([[recency, frequency, monetary]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict cluster
        cluster = int(model.predict(features_scaled)[0])
        
        # Get segment information
        segment_name = segment_info.loc[cluster, 'Segment']
        
        # Calculate similarity to cluster center
        cluster_center = model.cluster_centers_[cluster]
        distance = np.linalg.norm(features_scaled[0] - cluster_center)
        confidence = max(0, min(100, 100 - (distance * 10)))
        
        # Generate recommendations based on segment
        recommendations = get_recommendations(segment_name, recency, frequency, monetary)
        
        return jsonify({
            'success': True,
            'prediction': {
                'cluster_id': cluster,
                'segment_name': segment_name,
                'confidence': round(confidence, 2),
                'input_values': {
                    'recency': recency,
                    'frequency': frequency,
                    'monetary': monetary
                },
                'cluster_characteristics': {
                    'avg_recency': round(segment_info.loc[cluster, 'Recency'], 2),
                    'avg_frequency': round(segment_info.loc[cluster, 'Frequency'], 2),
                    'avg_monetary': round(segment_info.loc[cluster, 'Monetary'], 2)
                },
                'recommendations': recommendations
            }
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid input values: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500

def get_recommendations(segment_name, recency, frequency, monetary):
    """Generate personalized recommendations based on customer segment"""
    recommendations = []
    
    if segment_name == 'Champions':
        recommendations = [
            "Reward this customer with exclusive VIP benefits",
            "Ask for reviews and testimonials",
            "Engage them as brand ambassadors",
            "Offer early access to new products"
        ]
    elif segment_name == 'Loyal Customers':
        recommendations = [
            "Upsell higher value products",
            "Offer loyalty rewards program",
            "Ask for feedback and suggestions",
            "Provide personalized recommendations"
        ]
    elif segment_name == 'Big Spenders':
        recommendations = [
            "Market premium and exclusive products",
            "Offer personalized shopping experience",
            "Provide dedicated customer support",
            "Create exclusive membership tiers"
        ]
    elif segment_name == 'At Risk':
        recommendations = [
            "Send win-back campaigns",
            "Offer special discounts to re-engage",
            "Survey to understand issues",
            "Provide limited-time offers"
        ]
    elif segment_name == 'New Customers':
        recommendations = [
            "Provide onboarding support",
            "Offer welcome discounts",
            "Share educational content",
            "Build engagement through newsletters"
        ]
    else:  # Regular Customers
        recommendations = [
            "Maintain regular engagement",
            "Send personalized offers",
            "Encourage repeat purchases",
            "Build long-term relationship"
        ]
    
    return recommendations

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Predict segments for multiple customers
    Expected JSON input:
    {
        "customers": [
            {"recency": 10, "frequency": 5, "monetary": 1000},
            {"recency": 100, "frequency": 2, "monetary": 200}
        ]
    }
    """
    try:
        # Check if models are loaded
        if model is None or scaler is None:
            return jsonify({
                'success': False,
                'error': 'Models not loaded'
            }), 500
        
        # Get input data
        data = request.get_json()
        
        if not data or 'customers' not in data:
            return jsonify({
                'success': False,
                'error': 'No customer data provided'
            }), 400
        
        customers = data['customers']
        
        if not isinstance(customers, list) or len(customers) == 0:
            return jsonify({
                'success': False,
                'error': 'Customers must be a non-empty list'
            }), 400
        
        predictions = []
        
        for i, customer in enumerate(customers):
            try:
                # Extract features
                recency = float(customer['recency'])
                frequency = float(customer['frequency'])
                monetary = float(customer['monetary'])
                
                # Create feature array
                features = np.array([[recency, frequency, monetary]])
                
                # Scale and predict
                features_scaled = scaler.transform(features)
                cluster = int(model.predict(features_scaled)[0])
                segment_name = segment_info.loc[cluster, 'Segment']
                
                predictions.append({
                    'customer_index': i,
                    'cluster_id': cluster,
                    'segment_name': segment_name,
                    'input_values': {
                        'recency': recency,
                        'frequency': frequency,
                        'monetary': monetary
                    }
                })
            except Exception as e:
                predictions.append({
                    'customer_index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'total_customers': len(customers)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Customer Segmentation API Server")
    print("="*50)
    print("\nAvailable endpoints:")
    print("  GET  /                    - Home page")
    print("  GET  /api/health          - Health check")
    print("  GET  /api/segments        - Get all segments info")
    print("  POST /api/predict         - Predict single customer segment")
    print("  POST /api/batch-predict   - Predict multiple customer segments")
    print("\n" + "="*50)
    print("Starting server on http://127.0.0.1:3000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=3000)
