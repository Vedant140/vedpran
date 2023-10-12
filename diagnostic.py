import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
def diagnostic(df,target_col,feature_cols):
    Le=LabelEncoder()
    attribute_data_type = df[target_col].dtype
    is_numeric = pd.api.types.is_numeric_dtype(attribute_data_type)
    if is_numeric:
        print(f"{target_col} is a numeric attribute.")
    else:
        df[target_col]=Le.fit_transform(df[target_col])

    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform Linear Regression
    st.subheader("Linear Regression Analysis:")
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    y_pred = regression_model.predict(X_test)

    # Display regression results
    st.write("Coefficients:", regression_model.coef_)
    st.write("Intercept:", regression_model.intercept_)

    # Plot the regression results
    st.subheader("Regression Plot:")
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    st.pyplot(plt)

    # Evaluate the model
    st.subheader("Model Evaluation:")
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write("Mean Squared Error:", mse)
    st.write("R-squared:", r2)

    # Pair Plot
    if st.button("Generate Pair Plot"):
        st.subheader("Pair Plot")
        sns.pairplot(df, hue=target_col)
        st.pyplot(plt)

    # Histograms and Box Plots
    if st.button("Generate Histograms and Box Plots"):
        st.subheader("Histograms and Box Plots")
        for feature in feature_cols:
            st.write(f"### {feature} Distribution")
            fig, (ax1, ax2) = plt.subplots(1, 2)
            sns.histplot(df[feature], ax=ax1, kde=True)
            sns.boxplot(data=df, y=feature, x=target_col, ax=ax2)
            st.pyplot(plt)

    # Correlation Heatmap
    if st.button("Generate Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        corr_matrix = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
        st.pyplot(plt)

    # Scatter Matrix
    if st.button("Generate Scatter Matrix"):
        st.subheader("Scatter Matrix")
        pd.plotting.scatter_matrix(df[feature_cols], alpha=0.8, figsize=(10, 10), diagonal='hist')
        st.pyplot(plt)