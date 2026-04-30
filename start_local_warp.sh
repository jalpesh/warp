#!/bin/bash
set -e

echo "Starting Warp to Ollama Proxy..."

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies if needed
pip install -q fastapi uvicorn httpx

# Start the proxy in the background
echo "Starting proxy server on port 8080..."
python3 warp_ollama_proxy.py &
PROXY_PID=$!

# Ensure the proxy is killed when this script exits
trap "kill $PROXY_PID" EXIT

# Wait a second for the proxy to start
sleep 2

echo "Starting Warp with local proxy override..."
# Build and run the Warp CLI with the API override
WARP_SERVER_ROOT_URL="http://127.0.0.1:8080" cargo run --bin warp --release

echo "Warp exited."
