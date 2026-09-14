import streamlit as st
import json
import os

st.set_page_config(page_title="Coal Mine PPE Safety Dashboard", layout="wide")

st.title("⛑️ Coal Mine PPE Safety Dashboard")
st.caption("Live view of stored PPE violation alerts from alerts.json")

alerts_file = "alerts.json"

# Load alerts from the JSON file
if os.path.exists(alerts_file):
    with open(alerts_file, "r") as f:
        try:
            alerts = json.load(f)
        except json.JSONDecodeError:
            alerts = []
else:
    alerts = []

if not alerts:
    st.warning("No alerts found. Run your detection script to generate alerts.json.")
else:
    st.subheader(f"Total Alerts: {len(alerts)}")

    # Summary counts by alert type
    helmet_count = sum(1 for a in alerts if a.get("alert_type") == "HELMET VIOLATION")
    vest_count = sum(1 for a in alerts if a.get("alert_type") == "SAFETY VEST VIOLATION")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Alerts", len(alerts))
    col2.metric("Helmet Violations", helmet_count)
    col3.metric("Safety Vest Violations", vest_count)

    st.divider()

    # Display alerts as a table, most recent first
    display_alerts = list(reversed(alerts))
    table_data = [
        {
            "Alert Type": a.get("alert_type", "N/A"),
            "Confidence": a.get("confidence", "N/A"),
            "Timestamp": a.get("timestamp", "N/A"),
            "Location": a.get("location", "N/A"),
        }
        for a in display_alerts
    ]

    st.dataframe(table_data, use_container_width=True)