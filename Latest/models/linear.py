import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('agg')  # Use the 'agg' backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import ast

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