import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="RetailSense Agent", layout="wide")

st.title("🛒 RetailSense – Intelligent Inventory Agent")
st.write("AI-powered stock monitoring with automatic alerts")

# Load data
data = pd.read_csv("retailstock.csv")

# ML Model
X = data[["current_stock", "avg_sales"]]
y = data["current_stock"] - data["avg_sales"]

model = LinearRegression()
model.fit(X, y)

data["predicted_future_stock"] = model.predict(X)

# Agent logic
def agent_decision(row):
    if row["predicted_future_stock"] < 0:
        return "DEMAND SPIKE"
    elif row["current_stock"] < row["min_stock"]:
        return "LOW STOCK"
    else:
        return "OK"

data["agent_decision"] = data.apply(agent_decision, axis=1)

# Show table
st.subheader("📊 Inventory Status")
st.dataframe(data, use_container_width=True)

# Alerts
st.subheader("🚨 Agent Alerts")

alerts = data[data["agent_decision"] != "OK"]

if alerts.empty:
    st.success("✅ All products are sufficiently stocked")
else:
    for _, row in alerts.iterrows():
        if row["agent_decision"] == "LOW STOCK":
            st.warning(f"⚠️ LOW STOCK: {row['product']}")
        elif row["agent_decision"] == "DEMAND SPIKE":
            st.error(f"🔥 DEMAND SPIKE: {row['product']}")

st.caption("RetailSense Agent | ML + Automation + Streamlit")