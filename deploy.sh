#!/bin/bash


# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if Python 3 is installed
echo ""
echo "Checking prerequisites..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi
print_success "Python 3 is installed"

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
print_success "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
echo ""
echo "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_success "Virtual environment activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q -r requirements.txt
print_success "Dependencies installed"

# Check if dataset exists
echo ""
echo "Checking dataset..."
if [ ! -f "data/Online Retail.xlsx" ]; then
    print_warning "Dataset not found. Downloading..."
    mkdir -p data
    cd data
    wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx
    cd ..
    print_success "Dataset downloaded"
else
    print_success "Dataset found"
fi

# Run data preprocessing if needed
echo ""
echo "Checking processed data..."
if [ ! -f "data/rfm_data.csv" ]; then
    print_warning "Processed data not found. Running preprocessing..."
    python eda_preprocessing.py
    print_success "Data preprocessing completed"
else
    print_success "Processed data found"
fi

# Train model if needed
echo ""
echo "Checking trained model..."
if [ ! -f "models/kmeans_model.pkl" ]; then
    print_warning "Trained model not found. Training model..."
    python train_model.py
    print_success "Model training completed"
else
    print_success "Trained model found"
fi

# Display deployment options
echo ""
echo "=================================================="
echo "Deployment Options"
echo "=================================================="
echo ""
echo "1. Development Server (Flask)"
echo "2. Production Server (Gunicorn)"
echo "3. Exit"
echo ""
read -p "Select deployment option (1-3): " choice

case $choice in
    1)
        echo ""
        print_success "Starting Flask development server..."
        echo ""
        echo "Access the application at: http://127.0.0.1:3000"
        echo "Press CTRL+C to stop the server"
        echo ""
        python app.py
        ;;
    2)
        echo ""
        echo "Installing Gunicorn..."
        pip install -q gunicorn
        print_success "Gunicorn installed"
        echo ""
        print_success "Starting Gunicorn production server..."
        echo ""
        echo "Access the application at: http://127.0.0.1:3000"
        echo "Press CTRL+C to stop the server"
        echo ""
        gunicorn -w 4 -b 0.0.0.0:3000 app:app
        ;;
    3)
        print_success "Exiting..."
        exit 0
        ;;
    *)
        print_error "Invalid option"
        exit 1
        ;;
esac
