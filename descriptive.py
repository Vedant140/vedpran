# descriptive.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#Central tendency

def calculate_central_tendency_stats(data, target_attribute, selected_attribute):
    central_tendency_stats = data.groupby(target_attribute)[selected_attribute].describe()
    return central_tendency_stats

def plot_central_tendency_graph(data, target_attribute, selected_attribute):
    if target_attribute not in data.columns or selected_attribute not in data.columns:
        st.error("One or both of the specified columns do not exist in the DataFrame.")
        return
    data[target_attribute] = pd.to_numeric(data[target_attribute], errors='coerce')
    data[selected_attribute] = pd.to_numeric(data[selected_attribute], errors='coerce')

    if data[target_attribute].dtype != 'numeric' or data[selected_attribute].dtype != 'numeric':
        st.error("One or both of the specified columns could not be converted to numeric.")
        return

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=target_attribute, y=selected_attribute, data=data)
    plt.title(f'Central Tendency of {selected_attribute} by {target_attribute}')
    st.pyplot()

def calculate_variance_and_dispersion(data, selected_attribute):
    if selected_attribute not in data.columns:
        st.error("The specified column does not exist in the DataFrame.")
        return
    selected_data = data[selected_attribute]
    variance = selected_data.var()
    st.write(f"Variance of {selected_attribute}: {variance}")

    plt.figure(figsize=(8, 6))
    plt.hist(selected_data, bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {selected_attribute}')
    plt.xlabel(selected_attribute)
    plt.ylabel('Frequency')
    st.pyplot(plt)
def calculate_frequency_measures(data, selected_attribute):
    if selected_attribute not in data.columns:
        st.error("The specified column does not exist in the DataFrame.")
        return
    frequency_counts = data[selected_attribute].value_counts()
    st.write(f"Frequency Measures for {selected_attribute}:")
    st.write(frequency_counts)

    plt.figure(figsize=(10, 6))
    frequency_counts.plot(kind='bar')
    plt.title(f'Frequency Measures for {selected_attribute}')
    plt.xlabel(selected_attribute)
    plt.ylabel('Frequency')
    st.pyplot(plt)

