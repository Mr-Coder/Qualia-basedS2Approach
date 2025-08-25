#!/bin/bash

# Chakra UI Mathematical Reasoning Demo Launcher
# This script starts both the API server and the frontend

set -e

echo "🌟 Chakra UI Mathematical Reasoning Demo"
echo "======================================"
echo ""

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Please run this script from the modern-frontend-demo directory."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH."
    exit 1
fi

# Check if Node.js is available
if ! command -v npm &> /dev/null; then
    echo "❌ Error: Node.js/npm is not installed or not in PATH."
    exit 1
fi

echo "📦 Installing dependencies..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install flask flask-cors --break-system-packages 2>/dev/null || echo "Python dependencies already installed"

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm install

echo ""
echo "🚀 Starting services..."

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $(jobs -p) 2>/dev/null || true
    wait
    echo "✅ All services stopped."
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start real reasoning API server in background
echo "📡 Starting COT-DIR Mathematical Reasoning API server on http://localhost:5001"
python3 real_reasoning_api_server.py &
API_PID=$!

# Wait a moment for API server to start
sleep 2

# Start frontend development server
echo "🎨 Starting frontend on http://localhost:3000"
echo ""
echo "🌈 Chakra UI Mathematical Reasoning Interface:"
echo "   Frontend: http://localhost:3000"
echo "   API:      http://localhost:5001" 
echo "   Health:   http://localhost:5001/api/health"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Start Vite dev server
npm run dev

# If we get here, the frontend has stopped
cleanup