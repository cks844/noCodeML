import matplotlib
import numpy as np
matplotlib.use('agg')  # Use the 'agg' backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import ast
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
import pandas as pd
import seaborn as sns 
 

def perform_knn(file,target):
    df = pd.read_csv(file,index_col=False)
    y=df[target]
    target=[target]
    if 'id' in df.columns:
        target.append('id')
    X=df.drop(columns=target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    # Perform Linear Regression
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)

    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    img_str = base64.b64encode(image_stream.read()).decode('utf-8')
    return(accuracy,img_str,conf_matrix)