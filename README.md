# Web-Based Stock Price Projection using Deep Learning
## Overview
This project aims to build a web-based application that utilizes deep learning models to project stock prices. The application provides users with insights into potential future stock prices based on historical data. It leverages deep learning algorithms such as Long Short-Term Memory (LSTM) networks for time series forecasting.

## Features
User-friendly web interface for inputting stock symbols.
Integration with deep learning models for stock price prediction.
Visualization of predicted stock price trends using plotly-dash.
Historical data analysis to improve prediction accuracy.

## Technologies Used
Python
Django (web framework)
TensorFlow (deep learning library)
HTML/CSS/JavaScript (for the frontend)
Pandas, NumPy (data processing)
Matplotlib, Plotly Dash(data visualization)

## Installation
### Clone the repository:

git clone https://github.com/yourusername/investology-project.git
cd investology-project

### Create a virtual environment and activate it:

In cmd 
virtualenv "name_of_environment"
cd "name_of_environment"
Scripts\activate

### Install dependencies:

In cmd
pip install -r requirements.txt
python manage.py migrate

### Start the Django development server:

python manage.py runserver
Access the web application at http://localhost:8000 in your web browser.
