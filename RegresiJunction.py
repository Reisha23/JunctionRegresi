import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv('traffic.csv')

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

# Features and target variable
features = ['year', 'Month', 'Dat', 'Hour']
target = 'Vehicles'

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Linear Regression model (replace with your own model)
model = LinearRegression()
model.fit(X_train, y_train)

# Web App
st.title("Traffic Prediction Web App")

# Sidebar for user input
st.sidebar.header("User Input Features")
year = st.sidebar.slider("Select Year", min_value=int(data["year"].min()), max_value=int(data["year"].max()))
month = st.sidebar.slider("Select Month", min_value=int(data["Month"].min()), max_value=int(data["Month"].max()))
day = st.sidebar.slider("Select Day", min_value=int(data["Dat"].min()), max_value=int(data["Dat"].max()))
hour = st.sidebar.slider("Select Hour", min_value=int(data["Hour"].min()), max_value=int(data["Hour"].max()))

# Make prediction
user_input = np.array([[year, month, day, hour]])
prediction = model.predict(user_input)

# Display prediction
st.write(f"### Predicted Number of Vehicles:")
st.write(f"The predicted number of vehicles is: {prediction[0]:.2f}")

# Additional sections for data analysis (as in the previous code)

# Scatterplot for 'Day' vs 'Vehicles' with hue='Junction'
st.write("### Scatterplot: Day vs Vehicles (colored by Junction)")
fig_scatter = sns.scatterplot(x=data['Day'], y="Vehicles", data=data, hue='Junction')

# Save the figure
scatterplot_path = "scatterplot.png"
plt.savefig(scatterplot_path, format='png')

# Display the saved image
st.image(scatterplot_path, caption='Scatterplot: Day vs Vehicles (colored by Junction)', use_column_width=True)
