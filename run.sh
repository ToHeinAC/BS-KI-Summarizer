#!/bin/bash

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Check if running in development mode
if [ "$1" == "--dev" ]; then
    echo "Starting Streamlit app in development mode..."
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0
else
    echo "Starting Streamlit app..."
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
fi

echo "
Streamlit app is now running. Open your browser and navigate to:"
echo "http://localhost:8501"

echo "
To use the CLI version instead, run:"
echo "python document_summarizer.py ./files/your_file.pdf --model llama3.3:70b-instruct-q4_K_M --chunk-model llama3.2"
