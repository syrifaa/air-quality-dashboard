import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

df = pd.read_csv('./PRSA_Data_Aotizhongxin_20130301-20170228.csv')
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

st.sidebar.title('Main Features')
feature = st.sidebar.selectbox(
    'Select a feature to display:',
    ['Time Series Analysis', 'Distribution of PM10', 'Scatter Plot (TEMP vs O3)', 'Clustering of PM2.5 and PM10']
)

if feature == 'Time Series Analysis':
    st.title('Daily Pollution Levels Over Time')

    attributes = st.selectbox('Select an attribute:', ['SO2', 'NO2', 'PM2.5']) 
    att = df.resample('D', on='datetime')[attributes].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(att)
    plt.title(f'Daily {attributes} Levels Over Time')
    plt.xlabel('Date')
    plt.ylabel('µg/m³')
    plt.grid(True)
    st.pyplot(plt)

elif feature == 'Distribution of PM10':
    st.title('Distribution of PM10 Levels')

    plt.figure(figsize=(10, 5))
    sns.histplot(df['PM10'], bins=20, kde=True, color='blue')
    plt.title('Distribution of PM10 Levels')
    plt.xlabel('µg/m³')
    plt.ylabel('Frequency')
    st.pyplot(plt)

elif feature == 'Scatter Plot (TEMP vs O3)':
    st.title('Correlation between O3 and Temperature Levels')

    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='TEMP', y='O3', data=df, alpha=0.8)
    plt.title('Correlation between O3 and Temperature Levels')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('O3 (µg/m³)')
    st.pyplot(plt)

elif feature == 'Clustering of PM2.5 and PM10':
    st.title('Clustering Based on PM2.5 and PM10 Levels by Year')

    pollution_by_year = df.groupby('year').agg({
        'PM2.5': 'mean',
        'PM10': 'mean',
        'SO2': 'mean'
    }).reset_index()

    def classify_pollution(pm_value):
        if pm_value <= 50:
            return 'Low'
        else:
            return 'High'

    pollution_by_year['PM2.5_Category'] = pollution_by_year['PM2.5'].apply(classify_pollution)

    plt.figure(figsize=(10, 5))
    plt.scatter(pollution_by_year['PM2.5'], pollution_by_year['PM10'],
                c=pollution_by_year['PM2.5_Category'].apply(lambda x: {'Low': 0, 'High': 1}[x]),
                cmap='viridis')
    plt.colorbar(label='Pollution Category')
    plt.title('Clustering Based on PM2.5 and PM10 Levels by Year')
    plt.xlabel('PM2.5 (µg/m³)')
    plt.ylabel('PM10 (µg/m³)')
    st.pyplot(plt)