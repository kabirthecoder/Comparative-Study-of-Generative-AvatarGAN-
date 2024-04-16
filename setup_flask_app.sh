#!/bin/bash
# Activate the virtual environment
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development

# Run the Flask application
flask run
