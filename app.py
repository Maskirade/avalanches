#import packages
import streamlit as st
import pandas as pd
import re
import os
import time
import plotly.express as px
from textblob import TextBlob
from dotenv import load_dotenv

load_dotenv()

# Helper function to get dataset path
def get_dataset_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "data", "customer_reviews.csv")  
    return csv_path

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text

# Sentiment analysis function
def get_sentiment(text):
    text = clean_text(text)
    try:
        return TextBlob(text).sentiment.polarity 
    except Exception:
        return 0

st.title("Hello, Tenio and Galope")
st.write("Welcome to your first Streamlit app.")

# Layout two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Load Dataset"):
        try:
            csv_path = get_dataset_path()
            df = pd.read_csv(csv_path)

            st.session_state["df"] = df.copy()
            st.success("✅ Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("❌ Dataset not found. Make sure the CSV is in the /data/ folder.")
        except Exception as e:
            st.error(f"⚠️ Error loading dataset: {e}")

with col2:
    if st.button("🔍 Analyze Sentiment"):
        if "df" in st.session_state:
            with st.spinner("🔍 Analyzing sentiment... please wait"):
                df = st.session_state["df"].copy()
                time.sleep(3)

                # Validate required columns
                if "SUMMARY" not in df.columns or "PRODUCT" not in df.columns:
                    st.error("❌ The dataset must include 'SUMMARY' and 'PRODUCT' columns.")
                else:
                    # Compute sentiment score
                    df["SENTIMENT_SCORE"] = df["SUMMARY"].apply(get_sentiment)

                    # Assign sentiment labels
                    def label_sentiment(score):
                        if score > 0.2:
                            return "Positive"
                        elif score < -0.2:
                            return "Negative"
                        else:
                            return "Neutral"

                    df["SENTIMENT_LABEL"] = df["SENTIMENT_SCORE"].apply(label_sentiment)
                    st.session_state["df"] = df
                    st.success("✅ Sentiment analysis completed!")
        else:
            st.warning("⚠️ Please load the dataset first.")

# Display dataset and charts
if "df" in st.session_state:
    df = st.session_state["df"]

    st.subheader("🔍 Filter by Product")
    product_list = ["All Products"] + sorted(df["PRODUCT"].dropna().unique().tolist())
    product = st.selectbox("Choose a product", product_list)

    if product != "All Products":
        filtered_df = df[df["PRODUCT"] == product]
    else:
        filtered_df = df

    st.subheader(f"📁 Reviews for {product}")
    st.dataframe(filtered_df)

    # Only show charts if sentiment analysis has been done
    if "SENTIMENT_LABEL" in df.columns and "SENTIMENT_SCORE" in df.columns:
        st.subheader("📊 Average Sentiment Score by Product")
        grouped = filtered_df.groupby("PRODUCT")["SENTIMENT_SCORE"].mean().sort_values(ascending=False)
        fig1 = px.bar(
            x=grouped.index,
            y=grouped.values,
            color=grouped.values,
            color_continuous_scale="RdYlGn",
            labels={"x": "Product", "y": "Average Sentiment Score"},
            title="Average Sentiment per Product"
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("📈 Sentiment Distribution")
        sentiment_counts = filtered_df["SENTIMENT_LABEL"].value_counts()
        fig2 = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            color=sentiment_counts.index,
            labels={"x": "Sentiment", "y": "Count"},
            title="Sentiment Category Distribution"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("ℹ️ Please click*🔍 Analyze Sentiment** to generate charts.")