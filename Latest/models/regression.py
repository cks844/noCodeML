import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('agg')  # Use the 'agg' backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################

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

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################