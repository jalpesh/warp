#!/bin/bash
set -e

echo "Starting Warp to Ollama Proxy..."

# 1. Check if ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed."
    echo "Please install it from https://ollama.com/download or run:"
    echo "  brew install ollama"
    exit 1
fi

# 2. Check if ollama is running
if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null; then
    echo "❌ Ollama daemon is not running."
    echo "Please start the Ollama application or run 'ollama serve' in another terminal."
    exit 1
fi

# 3. Get available models
echo "🔍 Checking available Ollama models..."
MODELS=$(curl -s http://127.0.0.1:11434/api/tags | grep -o '"name":"[^"]*"' | cut -d'"' -f4)

TARGET_MODEL="llama3.2" # fallback default

if [ -z "$MODELS" ]; then
    echo "⚠️ No models found in Ollama. Pulling default model ($TARGET_MODEL)..."
    echo "This may take a few minutes depending on your internet connection."
    ollama pull "$TARGET_MODEL"
else
    # Auto-select the best coding model available, or fallback to whatever is installed
    if echo "$MODELS" | grep -q "qwen2.5-coder"; then
        TARGET_MODEL=$(echo "$MODELS" | grep "qwen2.5-coder" | head -n 1)
    elif echo "$MODELS" | grep -q "deepseek-coder"; then
        TARGET_MODEL=$(echo "$MODELS" | grep "deepseek-coder" | head -n 1)
    elif echo "$MODELS" | grep -q "llama3"; then
        TARGET_MODEL=$(echo "$MODELS" | grep "llama3" | head -n 1)
    else
        TARGET_MODEL=$(echo "$MODELS" | head -n 1)
    fi
    echo "✅ Found models. Auto-selected: $TARGET_MODEL"
fi

export OLLAMA_MODEL="$TARGET_MODEL"

# 4. Proxy Setup
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -q fastapi uvicorn httpx

echo "Starting proxy server on port 8080 (Model: $OLLAMA_MODEL)..."
python3 warp_ollama_proxy.py &
PROXY_PID=$!

trap "kill $PROXY_PID" EXIT
sleep 2

echo "🚀 Starting Warp with local proxy override..."
WARP_SERVER_ROOT_URL="http://127.0.0.1:8080" cargo run --bin warp --release

echo "Warp exited."
