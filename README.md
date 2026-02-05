#  Customer Segmentation ML Model Deployment

A complete machine learning project that performs customer segmentation using RFM (Recency, Frequency, Monetary) analysis on the Online Retail Dataset. The trained model is deployed as a web application with a Flask API backend and interactive HTML/CSS/JavaScript frontend.



##  Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional)

### Step-by-Step Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd /Deployment
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**
   ```bash
   # On Linux/Mac
   source venv/bin/activate
   
   # On Windows
   venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run data preprocessing** (if not already done)
   ```bash
   python eda_preprocessing.py
   ```

6. **Train the model** (if not already done)
   ```bash
   python train_model.py
   ```

##  Usage

### Starting the Application

#### Option 1: Using the Deployment Script (Recommended)

The easiest way to start the project is using the automated deployment script:

```bash
./deploy.sh
```

The script will:
- Check all prerequisites
- Set up the virtual environment
- Install dependencies
- Verify dataset and models
- Present deployment options (Development or Production server)

Select option 1 for development server or option 2 for production server (Gunicorn).

#### Option 2: Manual Start

1. **Activate the virtual environment** (if not already activated)
   ```bash
   source venv/bin/activate
   ```

2. **Run the Flask application**
   ```bash
   python app.py
   ```

3. **Access the web interface**
   - Open your browser and navigate to: `http://127.0.0.1:3000`

### Using the Web Interface

#### 1. Single Prediction
- Navigate to the "Single Prediction" tab
- Enter customer RFM values:
  - **Recency**: Days since last purchase (e.g., 30)
  - **Frequency**: Number of purchases (e.g., 5)
  - **Monetary**: Total spending (e.g., 1500.00)
- Click "Predict Segment"
- View the predicted segment and recommendations

#### 2. Batch Prediction
- Navigate to the "Batch Prediction" tab
- Enter JSON data for multiple customers:
  ```json
  {
    "customers": [
      {"recency": 10, "frequency": 5, "monetary": 1000},
      {"recency": 100, "frequency": 2, "monetary": 200}
    ]
  }
  ```
- Click "Predict Batch"
- View results for all customers

#### 3. View Segments
- Navigate to the "View Segments" tab
- Click "Load Segments"
- Explore all customer segments and their characteristics

## 🔌 API Documentation

### Base URL
```
http://127.0.0.1:3000/api
```

### Endpoints

#### 1. Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

#### 2. Get All Segments
```http
GET /api/segments
```

**Response:**
```json
{
  "success": true,
  "segments": [
    {
      "cluster_id": 0,
      "segment_name": "Loyal Customers",
      "avg_recency": 93.06,
      "avg_frequency": 3.90,
      "avg_monetary": 1548.68
    }
  ]
}
```

#### 3. Predict Single Customer Segment
```http
POST /api/predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "recency": 30,
  "frequency": 5,
  "monetary": 1500.00
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "cluster_id": 0,
    "segment_name": "Loyal Customers",
    "confidence": 87.5,
    "input_values": {
      "recency": 30,
      "frequency": 5,
      "monetary": 1500.0
    },
    "cluster_characteristics": {
      "avg_recency": 93.06,
      "avg_frequency": 3.90,
      "avg_monetary": 1548.68
    },
    "recommendations": [
      "Upsell higher value products",
      "Offer loyalty rewards program",
      "Ask for feedback and suggestions"
    ]
  }
}
```

#### 4. Batch Prediction
```http
POST /api/batch-predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "customers": [
    {"recency": 10, "frequency": 5, "monetary": 1000},
    {"recency": 100, "frequency": 2, "monetary": 200}
  ]
}
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "customer_index": 0,
      "cluster_id": 1,
      "segment_name": "Champions",
      "input_values": {...}
    }
  ],
  "total_customers": 2
}
```

##  Deployment Approaches

This project demonstrates **3 different deployment approaches**:

### 1. Local Development Server (Current)
- **Method**: Flask development server
- **Access**: http://127.0.0.1:3000
- **Use Case**: Development and testing
- **Command**: `./deploy.sh` or `python app.py`

### 2. Production Server (Gunicorn)
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:3000 app:app
```

### 3. Cloud Deployment Options

#### Option A: Heroku
```bash
# Create Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

#### Option B: AWS EC2
1. Set up EC2 instance
2. Install dependencies
3. Configure Nginx as reverse proxy
4. Use systemd for process management

#### Option C: Docker Container
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 3000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:3000", "app:app"]
```

