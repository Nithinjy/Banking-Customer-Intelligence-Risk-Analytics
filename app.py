import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
import streamlit as st
st.set_page_config(page_title="Banking Customer Intelligence Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("banking_customer_analytics_100k.csv")

df = load_data()

st.title("Banking Customer Risk, Churn & Profitability Intelligence Dashboard")

st.markdown(
    "This dashboard analyzes customer profitability, churn risk, loan default risk, "
    "digital banking behavior, complaints, cross-sell opportunity, and hidden business patterns."
)

st.sidebar.header("Filters")

selected_state = st.sidebar.multiselect(
    "State",
    sorted(df["State"].unique()),
    default=sorted(df["State"].unique())
)

selected_segment = st.sidebar.multiselect(
    "Customer Segment",
    sorted(df["Customer_Segment"].unique()),
    default=sorted(df["Customer_Segment"].unique())
)

selected_account = st.sidebar.multiselect(
    "Account Type",
    sorted(df["Account_Type"].unique()),
    default=sorted(df["Account_Type"].unique())
)

selected_risk = st.sidebar.multiselect(
    "Risk Segment",
    sorted(df["Risk_Segment"].unique()),
    default=sorted(df["Risk_Segment"].unique())
)

filtered_df = df[
    (df["State"].isin(selected_state)) &
    (df["Customer_Segment"].isin(selected_segment)) &
    (df["Account_Type"].isin(selected_account)) &
    (df["Risk_Segment"].isin(selected_risk))
].copy()

total_customers = filtered_df["Customer_ID"].nunique()
total_balance = filtered_df["Avg_Balance"].sum()
total_revenue = filtered_df["Monthly_Revenue"].sum()
total_profit = filtered_df["Monthly_Profit"].sum()

churn_rate = filtered_df["Churn_Flag"].mean() * 100 if len(filtered_df) else 0

loan_customers = filtered_df[filtered_df["Has_Loan"] == 1]
default_rate = loan_customers["Loan_Default_Flag"].mean() * 100 if len(loan_customers) else 0

k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric("Customers", f"{total_customers:,.0f}")
k2.metric("Total Balance", f"₹{total_balance/1e7:,.2f} Cr")
k3.metric("Monthly Revenue", f"₹{total_revenue/1e7:,.2f} Cr")
k4.metric("Monthly Profit", f"₹{total_profit/1e7:,.2f} Cr")
k5.metric("Churn Rate", f"{churn_rate:.2f}%")
k6.metric("Loan Default Rate", f"{default_rate:.2f}%")

st.markdown("---")

tabs = st.tabs([
    "Business Overview",
    "Profitability",
    "Risk & Default",
    "Churn",
    "Cross-Sell",
    "Hidden Patterns",
    "Recommendations",
    "Data"
])

with tabs[0]:
    st.header("Business Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Segment Distribution")
        segment_counts = filtered_df["Customer_Segment"].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(segment_counts.values, labels=segment_counts.index, autopct="%1.1f%%")
        ax.set_title("Customer Segment Distribution")
        st.pyplot(fig)

    with col2:
        st.subheader("Preferred Channel Distribution")
        channel_counts = filtered_df["Preferred_Channel"].value_counts().reset_index()
        channel_counts.columns = ["Preferred_Channel", "Customers"]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=channel_counts, x="Preferred_Channel", y="Customers", ax=ax)
        ax.set_title("Preferred Banking Channel")
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)

with tabs[1]:
    st.header("Profitability Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Profit by Customer Segment")

        profit_segment = (
            filtered_df.groupby("Customer_Segment")["Monthly_Profit"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=profit_segment, x="Customer_Segment", y="Monthly_Profit", ax=ax)
        ax.set_title("Monthly Profit by Segment")
        st.pyplot(fig)

        st.dataframe(profit_segment)

    with col2:
        st.subheader("Average Profit by Products Count")

        product_profit = (
            filtered_df.groupby("Products_Count")["Monthly_Profit"]
            .mean()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=product_profit, x="Products_Count", y="Monthly_Profit", marker="o", ax=ax)
        ax.set_title("Product Count vs Profit")
        st.pyplot(fig)

        st.dataframe(product_profit)

with tabs[2]:
    st.header("Risk & Loan Default Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Segment Distribution")

        risk_counts = filtered_df["Risk_Segment"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Segment", "Customers"]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=risk_counts, x="Risk_Segment", y="Customers", ax=ax)
        ax.set_title("Customer Risk Distribution")
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)

        st.dataframe(risk_counts)

    with col2:
        st.subheader("Default Rate by Risk Segment")

        loan_df = filtered_df[filtered_df["Has_Loan"] == 1]

        default_by_risk = (
            loan_df.groupby("Risk_Segment")["Loan_Default_Flag"]
            .mean()
            .mul(100)
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=default_by_risk, x="Risk_Segment", y="Loan_Default_Flag", ax=ax)
        ax.set_ylabel("Default Rate (%)")
        ax.set_title("Default Rate by Risk Segment")
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)

        st.dataframe(default_by_risk)

with tabs[3]:
    st.header("Churn Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Rate by Customer Segment")

        churn_segment = (
            filtered_df.groupby("Customer_Segment")["Churn_Flag"]
            .mean()
            .mul(100)
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=churn_segment, x="Customer_Segment", y="Churn_Flag", ax=ax)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Churn Rate by Segment")
        st.pyplot(fig)

        st.dataframe(churn_segment)

    with col2:
        st.subheader("Complaints vs Churn")

        complaint_pattern = (
            filtered_df.groupby("Complaints_3M")["Churn_Flag"]
            .mean()
            .mul(100)
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=complaint_pattern, x="Complaints_3M", y="Churn_Flag", marker="o", ax=ax)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Complaint Impact on Churn")
        st.pyplot(fig)

        st.dataframe(complaint_pattern)

with tabs[4]:
    st.header("Cross-Sell Opportunity Analysis")

    cross_sell_summary = filtered_df.groupby("Cross_Sell_Opportunity").agg(
        Customers=("Customer_ID", "nunique"),
        Avg_Balance=("Avg_Balance", "mean"),
        Avg_Credit_Score=("Credit_Score", "mean"),
        Avg_Products=("Products_Count", "mean"),
        Avg_Profit=("Monthly_Profit", "mean"),
        Churn_Rate=("Churn_Flag", "mean")
    ).reset_index()

    cross_sell_summary["Churn_Rate"] = cross_sell_summary["Churn_Rate"] * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=cross_sell_summary, x="Cross_Sell_Opportunity", y="Customers", ax=ax)
    ax.set_title("Customers by Cross-Sell Opportunity")
    st.pyplot(fig)

    st.dataframe(cross_sell_summary)

with tabs[5]:
    st.header("Hidden Pattern Analysis")

    st.subheader("Pattern 1: High Income but Low Balance")

    high_income_low_balance = filtered_df[
        (filtered_df["Annual_Income"] > filtered_df["Annual_Income"].quantile(0.75)) &
        (filtered_df["Avg_Balance"] < filtered_df["Avg_Balance"].quantile(0.25))
    ]

    st.write("These customers earn well but do not keep enough balance in this bank.")
    st.write(f"Customers found: {high_income_low_balance.shape[0]:,}")

    st.dataframe(
        high_income_low_balance[
            ["Customer_ID", "Annual_Income", "Avg_Balance", "Account_Type",
             "Customer_Segment", "Products_Count", "Monthly_Profit"]
        ].head(100)
    )

    st.subheader("Pattern 2: Good Credit Score but Recent Missed Payments")

    good_score_recent_miss = filtered_df[
        (filtered_df["Credit_Score"] >= 720) &
        (filtered_df["Missed_Payments_6M"] >= 2) &
        (filtered_df["Has_Loan"] == 1)
    ]

    st.write("These customers look good by credit score, but recent repayment behavior shows risk.")
    st.write(f"Customers found: {good_score_recent_miss.shape[0]:,}")

    st.dataframe(
        good_score_recent_miss[
            ["Customer_ID", "Credit_Score", "Missed_Payments_6M",
             "Outstanding_Loan", "Risk_Score", "Risk_Segment"]
        ].head(100)
    )

    st.subheader("Pattern 3: High Revenue but Low Profit")

    high_revenue_low_profit = filtered_df[
        (filtered_df["Monthly_Revenue"] > filtered_df["Monthly_Revenue"].quantile(0.75)) &
        (filtered_df["Monthly_Profit"] < filtered_df["Monthly_Profit"].quantile(0.25))
    ]

    st.write("These customers generate revenue, but high service cost reduces profit.")
    st.write(f"Customers found: {high_revenue_low_profit.shape[0]:,}")

    st.dataframe(
        high_revenue_low_profit[
            ["Customer_ID", "Monthly_Revenue", "Service_Cost", "Monthly_Profit",
             "Complaints_3M", "Transactions_30D", "Preferred_Channel"]
        ].head(100)
    )

with tabs[6]:
    st.header("Business Recommendations")

    high_risk_customers = filtered_df[
        filtered_df["Risk_Segment"].isin(["High Risk", "Very High Risk"])
    ]["Customer_ID"].nunique()

    high_churn_customers = filtered_df[
        filtered_df["Churn_Probability"] >= 0.50
    ]["Customer_ID"].nunique()

    high_cross_sell = filtered_df[
        filtered_df["Cross_Sell_Opportunity"] == "High"
    ]["Customer_ID"].nunique()

    loss_customers = filtered_df[
        filtered_df["Monthly_Profit"] < 0
    ]["Customer_ID"].nunique()

    st.write(f"1. **{high_risk_customers:,.0f} high-risk customers** need monitoring.")
    st.write(f"2. **{high_churn_customers:,.0f} high-churn customers** need retention campaigns.")
    st.write(f"3. **{high_cross_sell:,.0f} customers** are strong cross-sell opportunities.")
    st.write(f"4. **{loss_customers:,.0f} loss-making customers** need service cost optimization.")

    st.markdown(
        '''
        ### Final Actions

        - Improve complaint resolution to reduce churn.
        - Promote digital banking to reduce service cost.
        - Cross-sell to high-balance, low-product customers.
        - Monitor high-risk loan customers.
        - Protect HNI and Affluent customers.
        - Track profit, not just revenue.
        '''
    )

with tabs[7]:
    st.header("Filtered Data")
    st.dataframe(filtered_df.head(2000))