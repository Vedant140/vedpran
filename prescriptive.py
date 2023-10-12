import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
# Load the dataset
df = pd.read_csv('AllElectronics.csv')
scaler = StandardScaler()
def convert_to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def prescriptive_analysis():
    df['ratings'] = df['ratings'].apply(convert_to_float)
    # Streamlit app
    st.title('Amazon Electronics Products Analysis')

    # Display basic dataset information
    st.header('Dataset Information')
    st.write(f"Number of Rows: {df.shape[0]}")
    st.write(f"Number of Columns: {df.shape[1]}")

    # Display the first few rows of the dataset
    st.header('Sample Data')
    st.write(df.head())

    # Data Preprocessing
    # Convert the 'Price' column to a numerical format
    df['actual_price'] = df['actual_price'].str.replace('₹', '')
    df["actual_price"]=df["actual_price"].str.replace(',','').astype(float)

    # Visualizations
    st.header('Visualizations')

    # Distribution of product ratings
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x='ratings', ax=ax1)
    ax1.set_xlabel('Product Rating')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Product Ratings')
    st.pyplot(fig)

    # Distribution of product prices
    fig, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='actual_price', bins=20, kde=True, ax=ax2)
    ax2.set_xlabel('Product Price ($)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Product Prices')
    st.pyplot(fig)



    # Impute missing values (replace NaN with mean)
    df['ratings'].fillna(df['ratings'].mean(), inplace=True)
    # Filter products with high ratings and low prices
    recommended_products = df[(df['ratings'] >= 4.5) & (df['actual_price'] >=100000)]

    recommended_products = recommended_products[['name', 'actual_price', 'ratings']].sort_values(by=['ratings', 'actual_price'],
                                                                                                ascending=[False, True])
    st.write('Top Recommended Products:')
    st.write(recommended_products.head(5))

    fig, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='ratings', y='actual_price', ax=ax3)
    ax3.set_xlabel('Product Rating')
    ax3.set_ylabel('Product Price ($)')
    ax3.set_title('Relationship between Product Ratings and Prices by Category')
    st.pyplot(fig)



    # Conclusion - Continued
    st.header('Conclusions')

    # Conclusion 1: Top-rated and affordable products
    st.subheader('1. Top-rated and Affordable Products')
    st.write('Based on the analysis, consider recommending the following products to customers:')
    st.write(recommended_products.head(5))

    # Conclusion 2: Category insights
    st.subheader('2. Category Insights')
    st.write('The analysis shows that product ratings vary by category. Consider focusing on categories with higher average ratings for promotions or marketing efforts.')

    # Conclusion 3: Overall Summary
    st.subheader('3. Overall Summary')
    st.write('In summary, the analysis provides valuable insights into product ratings, prices, and categories. Use these insights to make informed decisions regarding product recommendations and marketing strategies.')
    # Conclusion
    st.header('Conclusion')
    st.write('Based on the analysis, consider recommending the top-rated and affordable products to customers.')

    # To run the Streamlit app, use the command: streamlit run your_app_name.py
