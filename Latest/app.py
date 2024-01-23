# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
import mysql.connector
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib
matplotlib.use('agg')  # Use the 'agg' backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import ast
app = Flask(__name__)

db_credentials = {
    'host': 'localhost',
    'user': 'root',
    'password': 'password',
    'database': 'NoCodeML'
}

# Helper function for linear regression
def perform_linear_regression(file):
    df = pd.read_csv(file)
    x = df.iloc[:, :-1].values.reshape(-1, 1)
    y = df.iloc[:, -1].values.reshape(-1, 1)

    # Perform Linear Regression
    model = LinearRegression()
    model.fit(x, y)

    # Create scatter plot and regression line
    plt.scatter(x, y)
    plt.plot(x, model.predict(x), color='red')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Save plot to BytesIO object
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    img_str = base64.b64encode(image_stream.read()).decode('utf-8')

    plt.close()

    return model, img_str



def authenticate_user(email, password):
    # Connect to the database
    db_connection = mysql.connector.connect(
        host=db_credentials['host'],
        user=db_credentials['user'],
        password=db_credentials['password'],
        database=db_credentials['database']
    )

    cursor = db_connection.cursor(dictionary=True)
    query = "SELECT * FROM users WHERE email = %s AND password = %s"
    cursor.execute(query, (email, password))
    user = cursor.fetchone()

    # Close the database connection
    db_connection.close()

    return user

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')

    user = authenticate_user(email, password)

    if user:
        return render_template('landing.html')  
    else:
        return render_template('sign-in.html', error='Invalid credentials')


# Main route
@app.route('/')
def index():
    return render_template('sign-in.html')

# Upload route
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    model, plot = perform_linear_regression(file)

    return render_template('linearreg.html', plot=plot, model=model)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    new_value = float(request.form['new_value'])  # Get the new value as a float
    # Create a new instance of LinearRegression
    model = LinearRegression()

    # Set the parameters for the new instance
    model.intercept_ = request.form['inter']
    model.intercept_=float(model.intercept_.strip('[]'))
    model.coef_ = request.form['coef']
    outer_list = ast.literal_eval(model.coef_)
    inner_list_floats = [float(item) for item in outer_list[0]]
    model.coef_ = np.array([inner_list_floats])
    # Reshape the new value to fit the model's input requirements
    new_value_reshaped = np.array([new_value]).reshape(-1, 1)

    # Use the trained model to predict the new value
    prediction = model.predict(new_value_reshaped)


    return render_template('linearreg.html', prediction=prediction, model=model, new_value=new_value)

app.run(debug=True, use_reloader=True, port=5004)
