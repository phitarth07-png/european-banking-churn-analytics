import os
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "European Banking — Customer Segmentation & Churn Analytics"
DATA_FILE = "European_Bank.csv"

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / DATA_FILE

REQUIRED_COLUMNS = [
    "Year",
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
]


# ============================================================
# DATA LOADING / VALIDATION / CLEANING
# ============================================================

def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            f"Keep '{DATA_FILE}' in the same folder as app_gradio.py."
        )

    try:
        data = pd.read_csv(DATA_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read {DATA_FILE}: {exc}"
        ) from exc

    missing_columns = [
        c for c in REQUIRED_COLUMNS
        if c not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset schema mismatch. Missing columns: "
            + ", ".join(missing_columns)
        )

    data = data[REQUIRED_COLUMNS].copy()

    # --------------------------------------------------------
    # TEXT CLEANING
    # --------------------------------------------------------

    for col in ["Surname", "Geography", "Gender"]:
        data[col] = (
            data[col]
            .astype("string")
            .str.strip()
        )

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    numeric_columns = [
        "Year",
        "CustomerId",
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]

    for col in numeric_columns:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # BOOLEAN / TARGET CLEANING
    # --------------------------------------------------------

    for col in [
        "HasCrCard",
        "IsActiveMember",
        "Exited",
    ]:
        data[col] = (
            data[col]
            .fillna(0)
            .clip(lower=0, upper=1)
            .round()
            .astype(int)
        )

    # --------------------------------------------------------
    # NUMERIC MISSING VALUES
    # --------------------------------------------------------

    numeric_fill_columns = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary",
    ]

    for col in numeric_fill_columns:

        if data[col].isna().all():
            data[col] = 0

        else:
            data[col] = data[col].fillna(
                data[col].median()
            )

    # --------------------------------------------------------
    # TEXT MISSING VALUES
    # --------------------------------------------------------

    for col in [
        "Geography",
        "Gender",
        "Surname",
    ]:

        if data[col].isna().all():
            data[col] = "Unknown"

        else:

            mode = data[col].mode(
                dropna=True
            )

            if not mode.empty:
                replacement = mode.iloc[0]
            else:
                replacement = "Unknown"

            data[col] = data[col].fillna(
                replacement
            )

    # --------------------------------------------------------
    # INTEGER FIELDS
    # --------------------------------------------------------

    for col in [
        "Year",
        "CustomerId",
        "Age",
        "Tenure",
        "NumOfProducts",
    ]:

        data[col] = (
            data[col]
            .round()
            .astype(int)
        )

    # --------------------------------------------------------
    # DUPLICATE REMOVAL
    # --------------------------------------------------------

    data = (
        data
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    # AGE SEGMENT

    data["AgeGroup"] = pd.cut(
        data["Age"],
        bins=[
            -np.inf,
            29,
            45,
            60,
            np.inf,
        ],
        labels=[
            "<30",
            "30–45",
            "46–60",
            "60+",
        ],
        right=True,
    ).astype(str)

    # CREDIT SCORE SEGMENT

    data["CreditScoreBand"] = pd.cut(
        data["CreditScore"],
        bins=[
            -np.inf,
            579,
            669,
            739,
            799,
            np.inf,
        ],
        labels=[
            "Poor",
            "Fair",
            "Good",
            "Very Good",
            "Excellent",
        ],
        right=True,
    ).astype(str)

    # TENURE SEGMENT

    data["TenureGroup"] = pd.cut(
        data["Tenure"],
        bins=[
            -np.inf,
            2,
            5,
            8,
            np.inf,
        ],
        labels=[
            "0–2 Years",
            "3–5 Years",
            "6–8 Years",
            "9+ Years",
        ],
        right=True,
    ).astype(str)

    # BALANCE SEGMENT

    data["BalanceSegment"] = pd.cut(
        data["Balance"],
        bins=[
            -np.inf,
            0,
            50000,
            100000,
            150000,
            np.inf,
        ],
        labels=[
            "Zero Balance",
            "Low",
            "Medium",
            "High",
            "Very High",
        ],
        include_lowest=True,
        right=True,
    ).astype(str)

    # PRODUCT SEGMENT

    data["ProductSegment"] = pd.cut(
        data["NumOfProducts"],
        bins=[
            -np.inf,
            1,
            2,
            np.inf,
        ],
        labels=[
            "1 Product",
            "2 Products",
            "3+ Products",
        ],
        right=True,
    ).astype(str)

    # ENGAGEMENT

    data["EngagementSegment"] = np.where(
        data["IsActiveMember"] == 1,
        "Active",
        "Inactive",
    )

    # --------------------------------------------------------
    # HIGH VALUE CUSTOMER
    # --------------------------------------------------------

    balance_threshold = data[
        "Balance"
    ].quantile(0.75)

    salary_threshold = data[
        "EstimatedSalary"
    ].quantile(0.75)

    data["HighValue"] = np.where(
        (
            data["Balance"]
            >= balance_threshold
        )
        |
        (
            data["EstimatedSalary"]
            >= salary_threshold
        ),
        "High Value",
        "Standard Value",
    )

    # --------------------------------------------------------
    # CHURN LABEL
    # --------------------------------------------------------

    data["ChurnLabel"] = data[
        "Exited"
    ].map({
        0: "Retained",
        1: "Churned",
    })

    return data


# ============================================================
# INITIAL DATA LOAD
# ============================================================

try:

    DF = load_dataset()
    LOAD_ERROR = None

except Exception as exc:

    DF = pd.DataFrame()
    LOAD_ERROR = str(exc)


# ============================================================
# GENERAL HELPERS
# ============================================================

def pct(value):
    return f"{value:.2f}%"


def safe_rate(
    numerator,
    denominator,
):
    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
        * 100
    )


def empty_figure(
    title,
    message="No data available for the selected filters.",
):

    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16),
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=430,
    )

    return fig


# ============================================================
# FILTER FUNCTION
# ============================================================

def apply_filters(
    geography="All",
    gender="All",
    age_group="All",
    credit_band="All",
    engagement="All",
    high_value="All",
):

    if DF.empty:
        return DF.copy()

    data = DF.copy()

    if geography != "All":
        data = data[
            data["Geography"]
            == geography
        ]

    if gender != "All":
        data = data[
            data["Gender"]
            == gender
        ]

    if age_group != "All":
        data = data[
            data["AgeGroup"]
            == age_group
        ]

    if credit_band != "All":
        data = data[
            data["CreditScoreBand"]
            == credit_band
        ]

    if engagement != "All":
        data = data[
            data["EngagementSegment"]
            == engagement
        ]

    if high_value != "All":
        data = data[
            data["HighValue"]
            == high_value
        ]

    return (
        data
        .reset_index(drop=True)
    )


# ============================================================
# FILTER SUMMARY
# ============================================================

def filter_summary(data):

    if data.empty:

        return (
            "### Current Selection\n"
            "No customers match the selected filters."
        )

    total = len(data)

    churned = int(
        data["Exited"].sum()
    )

    churn_rate = (
        data["Exited"].mean()
        * 100
    )

    return (
        "### Current Selection\n"
        f"**Customers:** {total:,}  |  "
        f"**Churned:** {churned:,}  |  "
        f"**Churn Rate:** {pct(churn_rate)}"
    )


# ============================================================
# KPI CALCULATIONS
# ============================================================

def calculate_kpis(data):

    if data.empty:

        return (
            "0",
            "0",
            "0%",
            "0%",
            "0.00x",
            "0%",
        )

    total = len(data)

    churned = int(
        data["Exited"].sum()
    )

    churn_rate = safe_rate(
        churned,
        total,
    )

    # --------------------------------------------------------
    # HIGH VALUE CHURN
    # --------------------------------------------------------

    high_value_data = data[
        data["HighValue"]
        == "High Value"
    ]

    high_value_churn_ratio = (
        safe_rate(
            int(
                high_value_data[
                    "Exited"
                ].sum()
            ),
            churned,
        )
        if churned > 0
        else 0
    )

    # --------------------------------------------------------
    # GEOGRAPHIC RISK
    # --------------------------------------------------------

    geography_rates = (
        data
        .groupby("Geography")[
            "Exited"
        ]
        .mean()
        .mul(100)
        .sort_values(
            ascending=False
        )
    )

    overall_rate = (
        data["Exited"].mean()
        * 100
    )

    geographic_risk_index = (

        geography_rates.max()
        / overall_rate

        if (
            overall_rate > 0
            and not geography_rates.empty
        )

        else 0
    )

    # --------------------------------------------------------
    # ENGAGEMENT DROP
    # --------------------------------------------------------

    if (
        data["IsActiveMember"]
        == 0
    ).any():

        inactive_churn = (
            data.loc[
                data["IsActiveMember"]
                == 0,
                "Exited",
            ]
            .mean()
            * 100
        )

    else:
        inactive_churn = 0

    if (
        data["IsActiveMember"]
        == 1
    ).any():

        active_churn = (
            data.loc[
                data["IsActiveMember"]
                == 1,
                "Exited",
            ]
            .mean()
            * 100
        )

    else:
        active_churn = 0

    engagement_drop = max(
        inactive_churn
        - active_churn,
        0,
    )

    return (
        f"{total:,}",
        f"{churned:,}",
        pct(churn_rate),
        pct(high_value_churn_ratio),
        f"{geographic_risk_index:.2f}x",
        pct(engagement_drop),
    )


# ============================================================
# GEOGRAPHY CHART
# ============================================================

def geography_chart(data):

    if data.empty:
        return empty_figure(
            "Geography-wise Churn"
        )

    temp = (
        data
        .groupby(
            "Geography",
            as_index=False,
        )
        .agg(
            ChurnRate=(
                "Exited",
                "mean",
            ),
            Customers=(
                "CustomerId",
                "count",
            ),
        )
    )

    temp["ChurnRate"] *= 100

    fig = px.bar(
        temp,
        x="Geography",
        y="ChurnRate",
        text=(
            temp["ChurnRate"]
            .round(1)
            .astype(str)
            + "%"
        ),
        hover_data={
            "Customers": True,
            "ChurnRate": ":.2f",
        },
        title="Churn Rate by Geography",
    )

    fig.update_layout(
        xaxis_title="Geography",
        yaxis_title="Churn Rate (%)",
        template="plotly_white",
        height=430,
    )

    return fig


# ============================================================
# AGE CHART
# ============================================================

def age_chart(data):

    if data.empty:
        return empty_figure(
            "Age-group Churn"
        )

    order = [
        "<30",
        "30–45",
        "46–60",
        "60+",
    ]

    temp = (
        data
        .groupby(
            "AgeGroup",
            observed=False,
        )["Exited"]
        .mean()
        .mul(100)
        .reindex(order)
        .reset_index()
    )

    temp.columns = [
        "AgeGroup",
        "ChurnRate",
    ]

    fig = px.bar(
        temp,
        x="AgeGroup",
        y="ChurnRate",
        text=(
            temp["ChurnRate"]
            .round(1)
            .astype(str)
            + "%"
        ),
        title="Churn Rate by Age Group",
    )

    fig.update_layout(
        xaxis_title="Age Group",
        yaxis_title="Churn Rate (%)",
        template="plotly_white",
        height=430,
    )

    return fig


# ============================================================
# TENURE CHART
# ============================================================

def tenure_chart(data):

    if data.empty:
        return empty_figure(
            "Tenure-group Churn"
        )

    order = [
        "0–2 Years",
        "3–5 Years",
        "6–8 Years",
        "9+ Years",
    ]

    temp = (
        data
        .groupby(
            "TenureGroup",
            observed=False,
        )["Exited"]
        .mean()
        .mul(100)
        .reindex(order)
        .reset_index()
    )

    temp.columns = [
        "TenureGroup",
        "ChurnRate",
    ]

    fig = px.bar(
        temp,
        x="TenureGroup",
        y="ChurnRate",
        text=(
            temp["ChurnRate"]
            .round(1)
            .astype(str)
            + "%"
        ),
        title="Churn Rate by Tenure",
    )

    fig.update_layout(
        xaxis_title="Tenure",
        yaxis_title="Churn Rate (%)",
        template="plotly_white",
        height=430,
    )

    return fig


# ============================================================
# CREDIT SCORE CHART
# ============================================================

def credit_chart(data):

    if data.empty:
        return empty_figure(
            "Credit-score-band Churn"
        )

    order = [
        "Poor",
        "Fair",
        "Good",
        "Very Good",
        "Excellent",
    ]

    temp = (
        data
        .groupby(
            "CreditScoreBand",
            observed=False,
        )["Exited"]
        .mean()
        .mul(100)
        .reindex(order)
        .reset_index()
    )

    temp.columns = [
        "CreditScoreBand",
        "ChurnRate",
    ]

    fig = px.bar(
        temp,
        x="CreditScoreBand",
        y="ChurnRate",
        text=(
            temp["ChurnRate"]
            .round(1)
            .astype(str)
            + "%"
        ),
        title="Churn Rate by Credit Score Band",
    )

    fig.update_layout(
        xaxis_title="Credit Score Band",
        yaxis_title="Churn Rate (%)",
        template="plotly_white",
        height=430,
    )

    return fig


# ============================================================
# ACTIVITY CHART
# ============================================================

def activity_chart(data):

    if data.empty:
        return empty_figure(
            "Active vs inactive"
        )

    temp = (
        data
        .groupby(
            "EngagementSegment",
            as_index=False,
        )["Exited"]
        .mean()
    )

    temp["Exited"] *= 100

    temp = temp.rename(
        columns={
            "Exited": "ChurnRate"
        }
    )

    fig = px.bar(
        temp,
        x="EngagementSegment",
        y="ChurnRate",
        text=(
            temp["ChurnRate"]
            .round(1)
            .astype(str)
            + "%"
        ),
        title="Churn Rate: Active vs Inactive",
    )

    fig.update_layout(
        xaxis_title="Engagement",
        yaxis_title="Churn Rate (%)",
        template="plotly_white",
        height=430,
    )

    return fig


# ============================================================
# CUSTOMER VALUE CHART
# ============================================================

def value_chart(data):

    if data.empty:
        return empty_figure(
            "High-value Customer Churn"
        )

    temp = (
        data
        .groupby(
            "HighValue",
            as_index=False,
        )["Exited"]
        .mean()
    )

    temp["Exited"] *= 100

    temp = temp.rename(
        columns={
            "Exited": "ChurnRate"
        }
    )

    fig = px.bar(
        temp,
        x="HighValue",
        y="ChurnRate",
        text=(
            temp["ChurnRate"]
            .round(1)
            .astype(str)
            + "%"
        ),
        title="Churn Rate: High-Value vs Standard",
    )

    fig.update_layout(
        xaxis_title="Customer Value Segment",
        yaxis_title="Churn Rate (%)",
        template="plotly_white",
        height=430,
    )

    return fig


# ============================================================
# PRODUCT COUNT CHART
# ============================================================

def products_chart(data):

    if data.empty:
        return empty_figure(
            "Product-count Churn"
        )

    temp = (
        data
        .groupby(
            "ProductSegment",
            as_index=False,
        )["Exited"]
        .mean()
    )

    temp["Exited"] *= 100

    temp = temp.rename(
        columns={
            "Exited": "ChurnRate"
        }
    )

    fig = px.bar(
        temp,
        x="ProductSegment",
        y="ChurnRate",
        text=(
            temp["ChurnRate"]
            .round(1)
            .astype(str)
            + "%"
        ),
        title="Churn Rate by Number of Products",
    )

    fig.update_layout(
        xaxis_title="Product Segment",
        yaxis_title="Churn Rate (%)",
        template="plotly_white",
        height=430,
    )

    return fig


# ============================================================
# GENDER CHART
# ============================================================

def gender_chart(data):

    if data.empty:
        return empty_figure(
            "Gender-wise Churn"
        )

    temp = (
        data
        .groupby(
            "Gender",
            as_index=False,
        )["Exited"]
        .mean()
    )

    temp["Exited"] *= 100

    temp = temp.rename(
        columns={
            "Exited": "ChurnRate"
        }
    )

    fig = px.bar(
        temp,
        x="Gender",
        y="ChurnRate",
        text=(
            temp["ChurnRate"]
            .round(1)
            .astype(str)
            + "%"
        ),
        title="Churn Rate by Gender",
    )

    fig.update_layout(
        xaxis_title="Gender",
        yaxis_title="Churn Rate (%)",
        template="plotly_white",
        height=430,
    )

    return fig


# ============================================================
# CUSTOMER OUTCOME CHART
# ============================================================

def churn_distribution_chart(data):

    if data.empty:
        return empty_figure(
            "Customer Outcome"
        )

    counts = (
        data["ChurnLabel"]
        .value_counts()
        .rename_axis("Outcome")
        .reset_index(
            name="Customers"
        )
    )

    fig = px.pie(
        counts,
        names="Outcome",
        values="Customers",
        hole=0.45,
        title="Customer Outcome Distribution",
    )

    fig.update_layout(
        template="plotly_white",
        height=430,
    )

    return fig


# ============================================================
# BALANCE VS SALARY SCATTER
# ============================================================

def balance_salary_scatter(data):

    if data.empty:
        return empty_figure(
            "Balance vs Estimated Salary"
        )

    plot_data = data.copy()

    plot_data[
        "ChurnStatus"
    ] = plot_data[
        "ChurnLabel"
    ]

    fig = px.scatter(
        plot_data,
        x="Balance",
        y="EstimatedSalary",
        color="ChurnStatus",
        hover_data=[
            "CustomerId",
            "Geography",
            "Age",
            "CreditScore",
            "NumOfProducts",
            "IsActiveMember",
        ],
        title="Balance vs Estimated Salary",
        opacity=0.65,
    )

    fig.update_layout(
        xaxis_title="Balance",
        yaxis_title="Estimated Salary",
        template="plotly_white",
        height=500,
    )

    return fig


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

def generate_insights(data):

    if data.empty:

        return (
            "## Business Insights\n"
            "No records match the current filters."
        )

    total = len(data)

    churned = int(
        data["Exited"].sum()
    )

    overall = (
        data["Exited"].mean()
        * 100
    )

    # --------------------------------------------------------
    # GEOGRAPHY
    # --------------------------------------------------------

    geo = (
        data
        .groupby("Geography")[
            "Exited"
        ]
        .mean()
        .mul(100)
        .sort_values(
            ascending=False
        )
    )

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    age = (
        data
        .groupby(
            "AgeGroup",
            observed=False,
        )["Exited"]
        .mean()
        .mul(100)
        .dropna()
        .sort_values(
            ascending=False
        )
    )

    # --------------------------------------------------------
    # TENURE
    # --------------------------------------------------------

    tenure = (
        data
        .groupby(
            "TenureGroup",
            observed=False,
        )["Exited"]
        .mean()
        .mul(100)
        .dropna()
        .sort_values(
            ascending=False
        )
    )

    # --------------------------------------------------------
    # ACTIVE VS INACTIVE
    # --------------------------------------------------------

    if (
        data["IsActiveMember"]
        == 1
    ).any():

        active_rate = (
            data.loc[
                data["IsActiveMember"]
                == 1,
                "Exited",
            ]
            .mean()
            * 100
        )

    else:
        active_rate = 0

    if (
        data["IsActiveMember"]
        == 0
    ).any():

        inactive_rate = (
            data.loc[
                data["IsActiveMember"]
                == 0,
                "Exited",
            ]
            .mean()
            * 100
        )

    else:
        inactive_rate = 0

    # --------------------------------------------------------
    # HIGH VALUE
    # --------------------------------------------------------

    high_value = data[
        data["HighValue"]
        == "High Value"
    ]

    if not high_value.empty:

        high_value_rate = (
            high_value[
                "Exited"
            ].mean()
            * 100
        )

    else:
        high_value_rate = 0

    lines = [

        "## Business Insights",

        (
            f"- **Overall churn:** "
            f"{pct(overall)} across "
            f"**{total:,} customers**."
        ),

        (
            f"- **Churned customers:** "
            f"**{churned:,}**."
        ),
    ]

    if not geo.empty:

        lines.append(
            f"- **Highest geographic churn:** "
            f"{geo.index[0]} at "
            f"**{pct(geo.iloc[0])}**."
        )

    if not age.empty:

        lines.append(
            f"- **Highest age-segment churn:** "
            f"{age.index[0]} at "
            f"**{pct(age.iloc[0])}**."
        )

    if not tenure.empty:

        lines.append(
            f"- **Highest tenure-segment churn:** "
            f"{tenure.index[0]} at "
            f"**{pct(tenure.iloc[0])}**."
        )

    lines.append(
        f"- **Inactive-member churn:** "
        f"{pct(inactive_rate)} versus "
        f"**{pct(active_rate)}** for active members."
    )

    lines.append(
        f"- **High-value customer churn:** "
        f"{pct(high_value_rate)}."
    )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    lines.extend([

        "",

        "## Recommended Business Actions",

        (
            "1. Prioritize retention campaigns "
            "for high-churn geographic and "
            "demographic segments."
        ),

        (
            "2. Build re-engagement campaigns "
            "for inactive customers using "
            "targeted offers and follow-ups."
        ),

        (
            "3. Create a dedicated retention "
            "watchlist for high-value customers "
            "showing churn characteristics."
        ),

        (
            "4. Review customers with low "
            "product engagement and identify "
            "relevant service or cross-sell opportunities."
        ),

        (
            "5. Monitor churn regularly by "
            "geography, age, tenure, credit score, "
            "activity and customer value."
        ),

    ])

    return "\n".join(lines)


# ============================================================
# GEOGRAPHIC SEGMENT SUMMARY TABLE
# ============================================================

def segment_table(data):

    if data.empty:

        return pd.DataFrame(
            columns=[
                "Segment",
                "Customers",
                "Churned",
                "Churn Rate",
                "Avg Balance",
                "Avg Salary",
            ]
        )

    temp = (
        data
        .groupby(
            "Geography",
            as_index=False,
        )
        .agg(
            Customers=(
                "CustomerId",
                "count",
            ),

            Churned=(
                "Exited",
                "sum",
            ),

            ChurnRate=(
                "Exited",
                "mean",
            ),

            AvgBalance=(
                "Balance",
                "mean",
            ),

            AvgSalary=(
                "EstimatedSalary",
                "mean",
            ),
        )
    )

    temp["ChurnRate"] = (
        temp["ChurnRate"]
        * 100
    ).round(2)

    temp["AvgBalance"] = (
        temp["AvgBalance"]
        .round(2)
    )

    temp["AvgSalary"] = (
        temp["AvgSalary"]
        .round(2)
    )

    temp = temp.rename(
        columns={
            "Geography": "Segment",
            "AvgBalance": "Avg Balance",
            "AvgSalary": "Avg Salary",
            "ChurnRate": "Churn Rate",
        }
    )

    return temp


# ============================================================
# HIGH VALUE CUSTOMER TABLE
# ============================================================

def high_value_table(data):

    columns = [
        "Customer ID",
        "Geography",
        "Age",
        "Credit Score",
        "Balance",
        "Estimated Salary",
        "Products",
        "Active Member",
        "Churn",
    ]

    if data.empty:
        return pd.DataFrame(
            columns=columns
        )

    temp = data[
        data["HighValue"]
        == "High Value"
    ].copy()

    if temp.empty:
        return pd.DataFrame(
            columns=columns
        )

    temp["Active Member"] = (
        temp["IsActiveMember"]
        .map({
            1: "Yes",
            0: "No",
        })
    )

    temp["Churn"] = (
        temp["Exited"]
        .map({
            1: "Churned",
            0: "Retained",
        })
    )

    temp = temp[
        [
            "CustomerId",
            "Geography",
            "Age",
            "CreditScore",
            "Balance",
            "EstimatedSalary",
            "NumOfProducts",
            "Active Member",
            "Churn",
        ]
    ]

    temp = temp.sort_values(
        [
            "Churn",
            "Balance",
        ],
        ascending=[
            True,
            False,
        ],
    )

    temp = temp.rename(
        columns={
            "CustomerId": "Customer ID",
            "CreditScore": "Credit Score",
            "EstimatedSalary": "Estimated Salary",
            "NumOfProducts": "Products",
        }
    )

    return temp.head(100)


# ============================================================
# MAIN DASHBOARD UPDATE
# ============================================================

def update_dashboard(
    geography,
    gender,
    age_group,
    credit_band,
    engagement,
    high_value,
):

    # --------------------------------------------------------
    # DATA ERROR
    # --------------------------------------------------------

    if LOAD_ERROR:

        error_md = (
            "## Dataset Error\n\n"
            f"```text\n"
            f"{LOAD_ERROR}\n"
            f"```\n\n"
            f"Make sure `{DATA_FILE}` is "
            f"in the same folder as `app_gradio.py`."
        )

        return (

            "0",

            "0",

            "0%",

            "0%",

            "0.00x",

            "0%",

            error_md,

            empty_figure(
                "Geography-wise Churn"
            ),

            empty_figure(
                "Age-group Churn"
            ),

            empty_figure(
                "Tenure-group Churn"
            ),

            empty_figure(
                "Credit-score-band Churn"
            ),

            empty_figure(
                "Active vs Inactive"
            ),

            empty_figure(
                "High-value Customer Churn"
            ),

            empty_figure(
                "Product-count Churn"
            ),

            empty_figure(
                "Gender-wise Churn"
            ),

            empty_figure(
                "Customer Outcome"
            ),

            empty_figure(
                "Balance vs Estimated Salary"
            ),

            pd.DataFrame(),

            pd.DataFrame(),
        )

    # --------------------------------------------------------
    # APPLY FILTERS
    # --------------------------------------------------------

    data = apply_filters(
        geography,
        gender,
        age_group,
        credit_band,
        engagement,
        high_value,
    )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    (
        total,
        churned,
        churn_rate,
        high_value_ratio,
        geo_index,
        engagement_drop,
    ) = calculate_kpis(data)

    # --------------------------------------------------------
    # INSIGHTS
    # --------------------------------------------------------

    insights = generate_insights(
        data
    )

    selected_summary = (
        filter_summary(data)
    )

    final_insights = (
        selected_summary
        + "\n\n"
        + insights
    )

    # --------------------------------------------------------
    # RETURN EVERYTHING
    # --------------------------------------------------------

    return (

        total,

        churned,

        churn_rate,

        high_value_ratio,

        geo_index,

        engagement_drop,

        final_insights,

        geography_chart(data),

        age_chart(data),

        tenure_chart(data),

        credit_chart(data),

        activity_chart(data),

        value_chart(data),

        products_chart(data),

        gender_chart(data),

        churn_distribution_chart(data),

        balance_salary_scatter(data),

        segment_table(data),

        high_value_table(data),
    )


# ============================================================
# DROPDOWN VALUES
# ============================================================

if not DF.empty:

    geography_choices = [
        "All"
    ] + sorted(
        DF["Geography"]
        .dropna()
        .unique()
        .tolist()
    )

    gender_choices = [
        "All"
    ] + sorted(
        DF["Gender"]
        .dropna()
        .unique()
        .tolist()
    )

    age_choices = [
        "All",
        "<30",
        "30–45",
        "46–60",
        "60+",
    ]

    credit_choices = [
        "All",
        "Poor",
        "Fair",
        "Good",
        "Very Good",
        "Excellent",
    ]

    engagement_choices = [
        "All",
        "Active",
        "Inactive",
    ]

    value_choices = [
        "All",
        "High Value",
        "Standard Value",
    ]

    dataset_status = (
        f"✅ **Dataset loaded successfully:** "
        f"{len(DF):,} customers | "
        f"{DF.shape[1]} source columns | "
        f"{int(DF['Exited'].sum()):,} churned customers"
    )

else:

    geography_choices = [
        "All"
    ]

    gender_choices = [
        "All"
    ]

    age_choices = [
        "All"
    ]

    credit_choices = [
        "All"
    ]

    engagement_choices = [
        "All"
    ]

    value_choices = [
        "All"
    ]

    dataset_status = (
        "❌ **Dataset could not be loaded.** "
        "See the error shown in the dashboard."
    )


# ============================================================
# CUSTOM CSS
# ============================================================

CUSTOM_CSS = """

.gradio-container {
    max-width: 1450px !important;
}

.hero {
    padding: 20px 24px;
    border-radius: 18px;
    background:
        linear-gradient(
            135deg,
            #111827,
            #1f2937
        );
    color: white;
    margin-bottom: 15px;
}

.hero h1 {
    margin-bottom: 5px;
}

.hero p {
    opacity: 0.90;
}

"""


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title=APP_TITLE,
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(),
) as demo:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.HTML(
        f"""
        <div class="hero">

            <h1>
                {APP_TITLE}
            </h1>

            <p>
                Segmentation-driven banking analytics
                dashboard covering churn, geography,
                demographics, engagement and customer value.
            </p>

        </div>
        """
    )

    # --------------------------------------------------------
    # DATASET STATUS
    # --------------------------------------------------------

    gr.Markdown(
        dataset_status
    )

    if LOAD_ERROR:

        gr.Markdown(
            f"""
            ### ⚠️ Dataset Setup Note

            `{LOAD_ERROR}`

            Keep `{DATA_FILE}` beside
            `app_gradio.py` and restart the application.
            """
        )

    # ========================================================
    # FILTER SECTION
    # ========================================================

    gr.Markdown(
        "## Dashboard Filters"
    )

    with gr.Row():

        with gr.Column():

            geography = gr.Dropdown(
                choices=geography_choices,
                value="All",
                label="Geography",
            )

        with gr.Column():

            gender = gr.Dropdown(
                choices=gender_choices,
                value="All",
                label="Gender",
            )

        with gr.Column():

            age_group = gr.Dropdown(
                choices=age_choices,
                value="All",
                label="Age Group",
            )

        with gr.Column():

            credit_band = gr.Dropdown(
                choices=credit_choices,
                value="All",
                label="Credit Score Band",
            )

        with gr.Column():

            engagement = gr.Dropdown(
                choices=engagement_choices,
                value="All",
                label="Engagement",
            )

        with gr.Column():

            high_value = gr.Dropdown(
                choices=value_choices,
                value="All",
                label="Customer Value",
            )

    # --------------------------------------------------------
    # REFRESH BUTTON
    # --------------------------------------------------------

    refresh = gr.Button(
        "Generate / Refresh Analysis",
        variant="primary",
        size="lg",
    )

    # ========================================================
    # KPI SECTION
    # ========================================================

    gr.Markdown(
        "## Key Performance Indicators"
    )

    with gr.Row():

        total_customers = gr.Textbox(
            label="Total Customers",
            value="0",
            interactive=False,
        )

        churned_customers = gr.Textbox(
            label="Churned Customers",
            value="0",
            interactive=False,
        )

        overall_churn = gr.Textbox(
            label="Overall Churn Rate",
            value="0%",
            interactive=False,
        )

        high_value_ratio = gr.Textbox(
            label="High-Value Churn Ratio",
            value="0%",
            interactive=False,
        )

        geographic_risk = gr.Textbox(
            label="Geographic Risk Index",
            value="0.00x",
            interactive=False,
        )

        engagement_drop_box = gr.Textbox(
            label="Engagement Drop Indicator",
            value="0%",
            interactive=False,
        )

    # ========================================================
    # EXECUTIVE INSIGHTS
    # ========================================================

    with gr.Tab(
        "Executive Insights"
    ):

        insights_output = gr.Markdown(
            """
            Use the filters and click
            **Generate / Refresh Analysis**.
            """
        )

    # ========================================================
    # GEOGRAPHY & DEMOGRAPHICS
    # ========================================================

    with gr.Tab(
        "Geography & Demographics"
    ):

        with gr.Row():

            geography_plot = gr.Plot(
                label="Geography"
            )

            age_plot = gr.Plot(
                label="Age"
            )

        with gr.Row():

            gender_plot = gr.Plot(
                label="Gender"
            )

            outcome_plot = gr.Plot(
                label="Customer Outcome"
            )

    # ========================================================
    # TENURE / CREDIT / ENGAGEMENT
    # ========================================================

    with gr.Tab(
        "Tenure, Credit & Engagement"
    ):

        with gr.Row():

            tenure_plot = gr.Plot(
                label="Tenure"
            )

            credit_plot = gr.Plot(
                label="Credit Score"
            )

        with gr.Row():

            activity_plot = gr.Plot(
                label="Engagement"
            )

            products_plot = gr.Plot(
                label="Products"
            )

    # ========================================================
    # CUSTOMER VALUE
    # ========================================================

    with gr.Tab(
        "Customer Value"
    ):

        with gr.Row():

            value_plot = gr.Plot(
                label="Customer Value"
            )

            salary_balance_plot = gr.Plot(
                label="Balance vs Salary"
            )

        gr.Markdown(
            """
            ### High-Value Customer Explorer

            This table highlights selected high-value
            customers including their balance, salary,
            products, activity and churn status.
            """
        )

        high_value_output = gr.Dataframe(
            headers=[
                "Customer ID",
                "Geography",
                "Age",
                "Credit Score",
                "Balance",
                "Estimated Salary",
                "Products",
                "Active Member",
                "Churn",
            ],
            interactive=False,
            wrap=True,
        )

    # ========================================================
    # SEGMENT SUMMARY
    # ========================================================

    with gr.Tab(
        "Segment Summary"
    ):

        segment_output = gr.Dataframe(
            headers=[
                "Segment",
                "Customers",
                "Churned",
                "Churn Rate",
                "Avg Balance",
                "Avg Salary",
            ],
            interactive=False,
            wrap=True,
        )

    # ========================================================
    # METHODOLOGY
    # ========================================================

    gr.Markdown(
        """
        ---

        ## Project Methodology

        **Data ingestion → validation → cleaning →
        segmentation → KPI analysis → churn comparison →
        high-value customer analysis → business recommendations**

        ### Core Segments

        - Geography
        - Age
        - Credit Score
        - Tenure
        - Balance
        - Product Count
        - Engagement
        - Customer Value

        ### Technology Stack

        **Python • Pandas • NumPy • Plotly • Gradio**

        ---
        """
    )

    # ========================================================
    # OUTPUT LIST
    # ========================================================

    outputs = [

        total_customers,

        churned_customers,

        overall_churn,

        high_value_ratio,

        geographic_risk,

        engagement_drop_box,

        insights_output,

        geography_plot,

        age_plot,

        tenure_plot,

        credit_plot,

        activity_plot,

        value_plot,

        products_plot,

        gender_plot,

        outcome_plot,

        salary_balance_plot,

        segment_output,

        high_value_output,
    ]

    # ========================================================
    # MAIN BUTTON EVENT
    # ========================================================

    refresh.click(

        fn=update_dashboard,

        inputs=[

            geography,

            gender,

            age_group,

            credit_band,

            engagement,

            high_value,
        ],

        outputs=outputs,
    )

    # ========================================================
    # AUTO UPDATE ON FILTER CHANGE
    # ========================================================

    for component in [

        geography,

        gender,

        age_group,

        credit_band,

        engagement,

        high_value,

    ]:

        component.change(

            fn=update_dashboard,

            inputs=[

                geography,

                gender,

                age_group,

                credit_band,

                engagement,

                high_value,
            ],

            outputs=outputs,
        )


# ============================================================
# PUBLIC HOSTING ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # Render automatically provides PORT.
    # Local fallback is 7860.

    port = int(
        os.environ.get(
            "PORT",
            "7860",
        )
    )

    # 0.0.0.0 is required so Render
    # can expose the application publicly.

    demo.launch(

        server_name="0.0.0.0",

        server_port=port,

        share=False,

        show_error=True,
    )