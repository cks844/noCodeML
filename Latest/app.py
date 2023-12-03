# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('agg')  # Use the 'agg' backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Helper function for linear regression
def perform_linear_regression(file):
    df = pd.read_csv(file)
    x = df.iloc[:,:-1].values.reshape(-1, 1)
    y = df.iloc[:,0:-1].values.reshape(-1, 1)

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

    return img_str, model

# Main route
@app.route('/')
def index():
    return render_template('landing.html')

# Upload route
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    plot, model = perform_linear_regression(file)

    return render_template('linearreg.html', plot=plot, model=model)


# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    new_value = float(request.form['new_value'])  # Get the new value as a float

    # Get the model parameters as a string and convert it to a dictionary
    model_params_str = request.form.get('model')
    model_params = eval(model_params_str) if model_params_str else {}

    # Create a new instance of LinearRegression
    model = LinearRegression()

    # Fit the model with dummy data to initialize its parameters
    dummy_x = [[0]]
    dummy_y = [0]
    model.fit(dummy_x, dummy_y)

    # Set the parameters for the model
    model.set_params(**model_params)

    # Reshape the new value to fit the model's input requirements
    new_value_reshaped = [[new_value]]

    # Use the trained model to predict the new value
    prediction = model.predict(new_value_reshaped)[0]

    return render_template('linearreg.html', prediction=prediction, model=model.get_params())


app.run(debug=True, use_reloader=True,port=5004)
