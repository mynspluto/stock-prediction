#!/bin/zsh

python3.12 -m venv ./api-server/api_server_env
source ./api-server/api_server_env/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn yfinance requests hdfs apache-airflow pydantic fastapi uvicorn confluent-kafka

# Function to run the API server
run_api_server() {
    echo "Starting FastAPI server..."
    python3.12 ./api-server/stock_prediction_api.py &
    API_PID=$!
    
    # Wait a moment for the server to start
    sleep 3
}

# Function to test the prediction endpoint
test_prediction() {
    echo "Testing prediction endpoint for ^IXIC..."
    curl -X GET "http://localhost:8000/predict/%5EIXIC"
    echo ""  # New line after curl output
}

# Function to cleanup
cleanup() {
    echo "Stopping API server..."
    kill $API_PID
    deactivate
}

# Trap to ensure cleanup on script exit
trap cleanup EXIT

# Run the server
run_api_server

# Test the prediction endpoint
test_prediction

# Keep the script running to maintain the server
echo "Server is running. Press Ctrl+C to stop."
wait