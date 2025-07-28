import streamlit as st
import pandas as pd
import numpy as np
import joblib

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Segmentation App")
st.write("Enter customer details to predict the segments.")

age = st.number_input("Age", min_value =18, max_value=100,value=35)
income = st.number_input("Income", min_value =0, max_value= 200000,value=5000)
total_spending =st.number_input("Total Spending (sum of purchase)", min_value=0, max_value=5000, value=1000)
num_web_purchase =st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)

num_store_purchase =st.number_input("Number of store Purchase", min_value=0, max_value=100, value= 10)
num_web_visits =st.number_input("Number of visits per Month", min_value=0, max_value=50, value=3)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

input_data = pd.DataFrame({
    "Age" : [age],
    "Income" : [income],
    "Total_Spending" : [total_spending],
    "NumWebPurchases" : [num_web_purchase],
    "NumStorePurchases" : [num_store_purchase],
    "NumWebVisitsMonth" : [num_web_visits],
    "Recency": [recency]
})

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):

    cluster = kmeans.predict(input_scaled)[0]

    st.success(f"Predicted Segment : Cluster {cluster}")

    st.write("""
Cluster 0: Budget Customers - This cluster has the lowest income & spending. They are also older and have not made a purchase recently.
             
Cluster 1: Premium Customers - This is the highest-earning cluster, and they are also the highest spenders. They are younger and have made a purchase recently.
             
Cluster 2: Senior Spenders - This cluster is the oldest and has the second-highest income and spending.
             
Cluster 3: Digital Buyers - This group has a high number of web purchases and a low number of in-store purchases. They have a high income and are of average age.
             
Cluster 4: Inactive Customers - This cluster has the lowest recency, meaning they have not made a purchase in a long time. They also have low income and spending.
             
Cluster 5: Frequent Buyers - This cluster has the highest number of purchases, both online and in-store. They have an average income and are of average age.""")