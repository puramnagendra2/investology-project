# Web-Based Stock Price Projection using Deep Learning
## Overview
This project aims to build a web-based application that utilizes deep learning models to project stock prices. The application provides users with insights into potential future stock prices based on historical data. It leverages deep learning algorithms such as Long Short-Term Memory (LSTM) networks for time series forecasting.

## Features
User-friendly web interface for inputting stock symbols.<br>
Integration with deep learning models for stock price prediction.<br>
Visualization of predicted stock price trends using plotly-dash.<br>
Historical data analysis to improve prediction accuracy.<br>

## Technologies Used<br>
Python<br>
Django (web framework)<br>
TensorFlow (deep learning library)<br>
HTML/CSS/JavaScript (for the frontend)<br>
Pandas, NumPy (data processing)<br>
Matplotlib, Plotly Dash(data visualization)<br>

## Installation
### Clone the repository:

git clone https://github.com/yourusername/investology-project.git<br>
cd investology-project<br>

### Create a virtual environment and activate it:

In cmd <br>
virtualenv "name_of_environment"<br>
cd "name_of_environment"<br>
Scripts\activate<br>

### Install dependencies:

In cmd<br>
pip install -r requirements.txt<br>
python manage.py migrate<br>

### Start the Django development server:

python manage.py runserver<br>
Access the web application at http://localhost:8000 in your web browser.
