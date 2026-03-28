import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Password Protection ---
APP_PASSWORD = os.getenv("APP_PASSWORD", "demo123")

password = st.text_input("Please enter a password to access the Flatbird Labs Revenue Attribution System", type="password")

if password != APP_PASSWORD:
    st.stop()

st.title("Flatbird Labs: Revenue Attribution System")

with st.expander("How to use this tool"):
    st.write("""
    1. Upload a CSV with columns: date, channel, spend, revenue  
    2. Review channel performance  
    3. Click 'Generate Insights' for recommendations  
    """)

st.markdown("""
Upload your marketing data and outcomes to identify which channels are driving revenue 
and where to optimize your budget for maximum impact.
""")

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")

client = None
if api_key:
    client = OpenAI(api_key=api_key)

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df)

    # Pivot data
    pivot_df = df.pivot_table(
        index="date",
        columns="channel",
        values="spend",
        aggfunc="sum"
    ).fillna(0)

    revenue_df = df.groupby("date")["revenue"].sum()
    model_df = pivot_df.merge(revenue_df, on="date")

    channels = list(pivot_df.columns)

    # Transform spend (diminishing returns)
    for ch in channels:
        model_df[f"{ch}_transformed"] = np.log1p(model_df[ch])

    X = model_df[[f"{ch}_transformed" for ch in channels]]
    y = model_df["revenue"]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    spend_totals = df.groupby("channel")["spend"].sum()

    st.subheader("Channel Spend")
    st.bar_chart(spend_totals)

    # Show coefficients
    st.subheader("Channel Contribution Analysis")
    coef_df = pd.DataFrame({
    "Channel": channels,
    "Performance Factor": model.coef_
    })

    # Normalize columns
    coef_df.columns = coef_df.columns.str.lower().str.strip()
    
    # Display table
    st.dataframe(coef_df)

    # ROI table
    st.subheader("Channel Performance")
    roi_df = pd.DataFrame({
    "Channel": channels,
    "Spend": [spend_totals[ch] for ch in channels],
    "Performance Factor": model.coef_
    })

    # Normalize column names (ADD THIS HERE)
    roi_df.columns = roi_df.columns.str.lower().str.strip()

    # Format spend (comes AFTER normalization)
    roi_df["spend"] = roi_df["spend"].map("${:,.0f}".format)
    
    # Winner / loser
    best_channel = coef_df.loc[coef_df["performance factor"].idxmax()]["channel"]
    worst_channel = coef_df.loc[coef_df["performance factor"].idxmin()]["channel"]

    st.success(f"Top Performing Channel: {best_channel}")
    st.warning(f"Underperforming Channel: {worst_channel}")

    # LLM Insights
    st.subheader("📊 Key Insights & Recommendations")

if st.button("Generate Insights"):
    if not client:
        st.warning("No OpenAI API key found. Insights are disabled.")
    else:
        insight_input = [
            {
                "channel": ch,
                "spend": float(spend_totals[ch]),
                "coefficient": float(model.coef_[i])
            }
            for i, ch in enumerate(channels)
        ]

        prompt = f"""
        You are a senior marketing strategist.

        Analyze this marketing data:
        {insight_input}

        1. Identify best and worst performing channels
        2. Recommend budget reallocations
        3. Estimate impact of changes

        Be specific and concise.
        """

        with st.spinner("Analyzing performance..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

        st.write(response.choices[0].message.content)