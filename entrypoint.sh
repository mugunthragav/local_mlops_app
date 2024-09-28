#!/bin/bash

# 1. Start MLflow UI
echo "Starting MLflow UI..."
mlflow ui --host 0.0.0.0 --port 5000 &

# 2. Run model configuration
echo "Running model configuration..."
python models/model_config.py

# 3. Run model training
echo "Running model training..."
python model_training.py

# 4. Promote the best model
echo "Promoting the best model..."
python promote_and_register.py

# 5. Start the Streamlit app to serve the model
echo "Starting Streamlit app..."
streamlit run app.py
