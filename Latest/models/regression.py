import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
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

def perform_svm(file,target):
    df = pd.read_csv(file,index_col=False)
    y=df[target]
    target=[target]
    if 'id' in df.columns:
        target.append('id')
    X=df.drop(columns=target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    model=SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)

    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    img_str = base64.b64encode(image_stream.read()).decode('utf-8')
    return(accuracy,img_str,conf_matrix)

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################