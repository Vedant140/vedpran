import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder

def perform_linear_regression_on_dataset(data):
    # Check if the DataFrame is empty
    if data.empty:
        st.error("The DataFrame is empty. Please provide data.")
        return

    # Extract the target column (last column)
    target_column = data.columns[-1]
    features = data.iloc[:, :-1]  # All columns except the last one
    Le=LabelEncoder()
    attribute_data_type = data[target_column].dtype
    is_numeric = pd.api.types.is_numeric_dtype(attribute_data_type)
    if is_numeric:
        print(f"{target_column} is a numeric attribute.")
    else:
        data[target_column]=Le.fit_transform(data[target_column])

    try:
        # Create and fit the linear regression model
        model = LinearRegression()
        model.fit(features, data[target_column])

        # Predict values using the model
        y_pred = model.predict(features)

        # Plot the original data and the regression line
        plt.figure(figsize=(8, 6))
        plt.scatter(data[target_column], y_pred, label="Original Data")
        plt.plot([min(data[target_column]), max(data[target_column])], [min(y_pred), max(y_pred)],
                 color='red', linestyle='-', linewidth=2, label="Regression Line")
        plt.title(f'Linear Regression: {target_column} vs Predicted')
        plt.xlabel(target_column)
        plt.ylabel("Predicted")
        plt.legend()
        st.pyplot(plt)

        # Display the coefficients of the linear regression equation
        st.write("Linear Regression Coefficients:")
        st.write(f"Intercept (b0): {model.intercept_}")
        st.write("Coefficients (b1, b2, ...):")
        for i, coef in enumerate(model.coef_):
            st.write(f"b{i + 1}: {coef}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

def create_neural_network_model(input_dim):
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', input_dim=input_dim))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1))  # Output layer with 1 neuron for regression
    return model

def ANN(data,target,attributes):
    st.title("Neural Networks Regression")
    X = data.iloc[:, :-1].values  # Features (all but the last column)
    y = data.iloc[:, -1].values
    st.subheader("Select the train-test split ratio:")
    split_ratio = st.slider("Train-Test Split Ratio", 0.1, 0.9, 0.7, 0.1)
    if data is not None:
        # Perform train-test split
        split_index = int(len(X) * split_ratio)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Create a neural network model
        model = create_neural_network_model(X_train.shape[1])

        # Compile the model with a loss function for regression
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        st.subheader("Training the Neural Network Model")
        model.fit(X_train, y_train, epochs=100, batch_size=32)

        # Evaluate the model
        loss = model.evaluate(X_test, y_test)
        st.subheader("Model Evaluation")
        st.write("Mean Squared Error on test data:", loss)
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []
        st.subheader("Training the Neural Network Model")
        for epoch in range(100):
            history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
            train_loss_history.append(history.history['loss'][0])
            val_loss_history.append(history.history['val_loss'][0])

            # Calculate dummy accuracy (replace with actual evaluation metric)
            train_acc_history.append(0.9)
            val_acc_history.append(0.85)

        # Plot training and validation loss
        st.subheader("Loss Curves")
        st.line_chart(pd.DataFrame({'Train Loss': train_loss_history, 'Validation Loss': val_loss_history}))

        # Plot training and validation accuracy (dummy values)
        st.subheader("Accuracy Curves")
        st.line_chart(pd.DataFrame({'Train Accuracy': train_acc_history, 'Validation Accuracy': val_acc_history}))

        st.subheader("Predict with the trained model")
        input_features = []


import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_iris_classification_model():
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a random forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model
def irispred():

    # Train the classification model
    model = train_iris_classification_model()


    sepal_length = st.number_input("Sepal Length:")
    sepal_width = st.number_input("Sepal Width:")
    petal_length = st.number_input("Petal Length:")
    petal_width = st.number_input("Petal Width:")

    if st.button("Predict"):
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        predicted_class = model.predict(input_data)[0]

        # Map class indices to species names
        iris = load_iris()
        species_names = iris.target_names
        predicted_species = species_names[predicted_class]

        st.write(f"Predicted Species: {predicted_species}")

