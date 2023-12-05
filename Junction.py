import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('traffic.csv')

# Display the first few rows of the data
st.write("## Traffic Data")
st.dataframe(data.head())

# Display the shape of the data
st.write("### Data Shape")
st.write(data.shape)

# Display null values and data info
st.write("### Null Values")
st.write(data.isnull())
st.write("### Data Info")
st.write(data.info())

# Display unique values, basic statistics, and statistics for object columns
st.write("### Unique Values")
st.write(data.nunique())
st.write("### Data Statistics")
st.write(data.describe())
st.write("### Object Columns Statistics")
st.write(data.describe(include='object'))

# Convert 'DateTime' to datetime and extract year, month, day, hour, and day of the week
data['DateTime'] = pd.to_datetime(data['DateTime'])
data["year"] = data['DateTime'].dt.year
data["Month"] = data['DateTime'].dt.month
data["Dat"] = data['DateTime'].dt.day
data["Hour"] = data['DateTime'].dt.hour
data["Day"] = data['DateTime'].dt.strftime("%A")

# Display the updated data
st.write("### Updated Data")
st.dataframe(data)

# Display duplicated rows
st.write("### Duplicated Rows")
st.write("Number of duplicated rows:", data.duplicated().sum())

# Plot line graphs for each date-related feature
date_hr = data[['Day', 'year', 'Month', 'Dat', 'Hour']]
for i in date_hr:
    st.write(f"### Lineplot: {i} vs Vehicles")
    plt.figure(figsize=(15, 6))
    sns.lineplot(x=date_hr[i], y='Vehicles', data=data, hue='Junction')
    plt.title(f'{i} vs Vehicles')
    plt.xlabel(i)
    plt.ylabel('Vehicles')
    plt.legend()
    plt.xticks(rotation=90)
    st.pyplot()

# Countplot for 'Junction'
st.write("### Countplot: Junction")
st.write(sns.countplot(x=data['Junction'], data=data))
st.pyplot()

# Scatterplot for 'Day' vs 'Vehicles' with hue='Junction'
st.write("### Scatterplot: Day vs Vehicles (colored by Junction)")
st.write(sns.scatterplot(x=data['Day'], y="Vehicles", data=data, hue='Junction'))
st.pyplot()
