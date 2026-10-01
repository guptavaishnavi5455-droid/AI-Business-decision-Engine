import streamlit as st
import pandas as pd
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# =================================================
# PAGE SETTINGS
# =================================================

st.set_page_config(
    page_title="AI Business Decision Engine",
    page_icon="🤖",
    layout="wide"
)


# =================================================
# LOAD DATASET
# =================================================

data = pd.read_csv("business_data.csv")

X = data[
    [
        "investment",
        "revenue",
        "expenses",
        "demand",
        "competition",
        "scalability"
    ]
]

y = data["viability"]


# =================================================
# TRAIN / TEST MODEL
# =================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

model_accuracy = accuracy_score(
    y_test,
    y_test_pred
)


# =================================================
# TITLE
# =================================================

st.set_page_config(
    page_title="AI Business Decision Engine",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Business Decision Engine")

st.write(
    "An AI-powered system for analyzing business ideas "
    "using financial, market and scalability factors."
)

st.caption(
    "Decision support prototype using rule-based analysis "
    "and a Random Forest machine-learning model."
)
st.write(
    "Analyze your business idea using financial, "
    "market and scalability factors."
)

st.info(
    "This prototype combines rule-based business analysis "
    "with a Random Forest machine-learning model."
)

st.caption(
    "ML model trained using simulated/demo business data."
)

st.divider()
# =========================================
# PROJECT INFORMATION SIDEBAR
# =========================================

with st.sidebar:
    st.header("🤖 About This Project")

    st.write(
        "AI Business Decision Engine helps analyze "
        "business ideas using financial, market and "
        "scalability factors."
    )

    st.divider()

    st.subheader("🛠️ Technologies")

    st.write("• Python")
    st.write("• Streamlit")
    st.write("• Pandas")
    st.write("• Scikit-learn")
    st.write("• Random Forest")
    st.write("• Plotly")
    st.write("• ReportLab")

    st.divider()

    st.caption(
        "This is a decision-support prototype "
        "using simulated business data."
    )


# =================================================
# BUSINESS INFORMATION
# =================================================
st.subheader("🔄 How the AI Decision Engine Works")

st.write(
    "Enter your business details → "
    "Financial analysis → "
    "Business scoring → "
    "AI/ML prediction → "
    "Risk analysis → "
    "Recommendations → "
    "Final decision"
)

st.divider()
st.header("🏢 1. Business Information")

business_name = st.text_input(
    "Business Idea Name",
    placeholder="Example: Online Grocery Delivery"
)

industry = st.selectbox(
    "Business Category",
    [
        "Technology",
        "Food & Beverage",
        "Retail",
        "Healthcare",
        "Education",
        "Finance",
        "Agriculture",
        "E-commerce",
        "Other"
    ]
)

target_customers = st.text_input(
    "Target Customers",
    placeholder="Example: College students"
)


# =================================================
# FINANCIAL INFORMATION
# =================================================

st.header("💰 2. Financial Information")

col1, col2, col3 = st.columns(3)

with col1:

    investment = st.number_input(
        "Initial Investment (₹)",
        min_value=0,
        value=100000,
        step=10000
    )

with col2:

    revenue = st.number_input(
        "Expected Monthly Revenue (₹)",
        min_value=0,
        value=50000,
        step=5000
    )

with col3:

    expenses = st.number_input(
        "Expected Monthly Expenses (₹)",
        min_value=0,
        value=30000,
        step=5000
    )


# =================================================
# MARKET FACTORS
# =================================================

st.header("📈 3. Market Factors")

col1, col2, col3 = st.columns(3)

with col1:

    demand = st.select_slider(
        "Market Demand",
        options=["Low", "Medium", "High"],
        value="Medium"
    )

with col2:

    competition = st.select_slider(
        "Competition",
        options=["Low", "Medium", "High"],
        value="Medium"
    )

with col3:

    scalability = st.select_slider(
        "Scalability",
        options=["Low", "Medium", "High"],
        value="Medium"
    )


# Convert market factors to numerical values

demand_score = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}[demand]

competition_score = {
    "Low": 3,
    "Medium": 2,
    "High": 1
}[competition]

scalability_score = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}[scalability]


# =================================================
# ANALYZE BUTTON
# =================================================

st.divider()

analyze = st.button(
    "🚀 Analyze Business Idea",
    use_container_width=True
)


if analyze:

    if not business_name:

        st.warning(
            "Please enter your business idea name."
        )

    else:

        # =========================================
        # FINANCIAL CALCULATIONS
        # =========================================

        monthly_profit = revenue - expenses

        if revenue > 0:

            profit_margin = (
                monthly_profit / revenue
            ) * 100

        else:

            profit_margin = 0


        if monthly_profit > 0:

            break_even_months = (
                investment / monthly_profit
            )

        else:

            break_even_months = None


        # =========================================
        # RULE-BASED BUSINESS SCORE
        # =========================================

        score = 0

        if monthly_profit > 0:

            score += 25

        if profit_margin >= 20:

            score += 20

        elif profit_margin >= 10:

            score += 10


        if demand_score == 3:

            score += 20

        elif demand_score == 2:

            score += 10


        if competition_score == 3:

            score += 15

        elif competition_score == 2:

            score += 8


        if scalability_score == 3:

            score += 20

        elif scalability_score == 2:

            score += 10


        score = min(score, 100)


        # =========================================
        # ML PREDICTION
        # =========================================
# =========================================
        # ML BUSINESS VIABILITY PREDICTION
        # =========================================

        ml_input = pd.DataFrame({
            "investment": [investment],
            "revenue": [revenue],
            "expenses": [expenses],
            "demand": [demand_score],
            "competition": [competition_score],
            "scalability": [scalability_score]
        })

        ml_prediction = model.predict(ml_input)[0]

        ml_probability = model.predict_proba(ml_input)[0]

        ml_confidence = max(ml_probability) * 100
# =========================================
        # AI/ML PREDICTION DISPLAY
        # =========================================

        st.subheader("🤖 AI/ML Business Viability Prediction")

        if ml_prediction == 1:
            ml_result = "High Viability"
        else:
            ml_result = "Lower Viability"

        st.success(
            f"AI/ML Prediction: {ml_result}"
        )

        st.info(
            f"Model Confidence: {ml_confidence:.2f}%"
        )
# =========================================
        # RISK FACTORS
        # =========================================

        st.subheader("⚠️ Risk Factors")

        risks = []

        if monthly_profit <= 0:
            risks.append("Monthly expenses are equal to or higher than revenue.")

        if profit_margin < 20:
            risks.append("Profit margin is relatively low.")

        if demand_score == 1:
            risks.append("Market demand is low.")

        if competition_score == 1:
            risks.append("Competition level is high.")

        if scalability_score == 1:
            risks.append("Scalability potential is limited.")

        if risks:
            for risk in risks:
                st.warning(risk)
        else:
            st.success("No major risk factors identified from the provided inputs.")
# =========================================
        # PERSONALIZED RECOMMENDATIONS
        # =========================================

        st.subheader("💡 Personalized Recommendations")

        recommendations = []

        if monthly_profit <= 0:
            recommendations.append(
                "Reduce operating expenses or increase monthly revenue."
            )

        if profit_margin < 20:
            recommendations.append(
                "Improve pricing or reduce costs to increase profit margin."
            )

        if demand_score == 1:
            recommendations.append(
                "Research customer needs and improve the product or service."
            )

        if competition_score == 1:
            recommendations.append(
                "Differentiate the business through better pricing, quality, or service."
            )

        if scalability_score == 1:
            recommendations.append(
                "Consider technology, automation, or a scalable business model."
            )

        if not recommendations:
            recommendations.append(
                "The business shows healthy financial and market indicators. "
                "Focus on customer growth and sustainable expansion."
            )
        for recommendation in recommendations:
                   st.info("💡 " + recommendation)        
# =========================================
        # FINAL BUSINESS DECISION
        # =========================================

        st.subheader("🎯 Final Business Decision")

        if score >= 75:
            decision = "Business Idea Shows Strong Potential"
            st.success(f"✅ {decision}")

        elif score >= 50:
            decision = "Business Idea Needs Improvement"
            st.warning(f"⚠️ {decision}")

        else:
            decision = "Business Idea Has High Risk"
            st.error(f"❌ {decision}")

        st.write(
            "This decision is based on the financial performance, "
            "market demand, competition and scalability factors provided."
        )

        # =========================================
        # BUSINESS SUMMARY
        # =========================================

        st.subheader("📋 Business Summary")

        st.write(f"**Business Idea:** {business_name}")
        st.write(f"**Monthly Revenue:** ₹{revenue:,.0f}")
        st.write(f"**Monthly Expenses:** ₹{expenses:,.0f}")
        st.write(f"**Monthly Profit:** ₹{monthly_profit:,.0f}")
        if break_even_months is not None:
            st.write(
                f"**Break-even Period:** {break_even_months:.1f} months"
            )
        else:
            st.write(
                "**Break-even Period:** Not reached"
            )
        st.write(f"**Profit Margin:** {profit_margin:.2f}%")
        st.write(f"**Business Score:** {score}/100")
        st.write(f"**AI/ML Prediction:** {ml_result}")
        st.write(f"**Final Decision:** {decision}")
# =========================================
        # DOWNLOAD BUSINESS REPORT
        # =========================================

        report = f"""
AI BUSINESS DECISION ENGINE
============================

Business Idea: {business_name}

FINANCIAL ANALYSIS
------------------
Monthly Revenue: ₹{revenue:,.0f}
Monthly Expenses: ₹{expenses:,.0f}
Monthly Profit: ₹{monthly_profit:,.0f}
Profit Margin: {profit_margin:.2f}%
Break-even Period: {break_even_months:.1f} months

BUSINESS ANALYSIS
-----------------
Business Score: {score}/100
AI/ML Prediction: {ml_result}
AI/ML Confidence: {ml_confidence:.2f}%

FINAL DECISION
--------------
{decision}

This report was generated by the AI Business Decision Engine.
"""
# =========================================
        # CREATE PDF REPORT
        # =========================================

        pdf_buffer = BytesIO()

        pdf = canvas.Canvas(pdf_buffer, pagesize=A4)

        pdf.setTitle("AI Business Analysis Report")

        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(50, 800, "AI BUSINESS DECISION ENGINE")

        pdf.setFont("Helvetica", 11)

        y = 770

        pdf.drawString(50, y, f"Business Idea: {business_name}")
        y -= 30

        pdf.drawString(50, y, "FINANCIAL ANALYSIS")
        y -= 20

        pdf.drawString(70, y, f"Monthly Revenue: Rs. {revenue:,.0f}")
        y -= 18
        pdf.drawString(70, y, f"Monthly Expenses: Rs. {expenses:,.0f}")
        y -= 18
        pdf.drawString(70, y, f"Monthly Profit: Rs. {monthly_profit:,.0f}")
        y -= 18
        pdf.drawString(70, y, f"Profit Margin: {profit_margin:.2f}%")
        y -= 18

        if break_even_months is not None:
            pdf.drawString(
                70, y,
                f"Break-even Period: {break_even_months:.1f} months"
            )
            y -= 18

        y -= 15
        pdf.drawString(50, y, "BUSINESS ANALYSIS")
        y -= 20

        pdf.drawString(70, y, f"Business Score: {score}/100")
        y -= 18
        pdf.drawString(70, y, f"AI/ML Prediction: {ml_result}")
        y -= 18
        pdf.drawString(70, y, f"AI/ML Confidence: {ml_confidence:.2f}%")
        y -= 30

        pdf.drawString(50, y, "FINAL DECISION")
        y -= 20

        pdf.drawString(70, y, decision)

        y -= 40

        pdf.drawString(
            50,
            y,
            "Generated by AI Business Decision Engine"
        )

        pdf.save()

        pdf_buffer.seek(0) 

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer,
file_name="business_analysis_report.pdf",
            mime="application/pdf"
        )

        st.download_button(
            label="📥 Download Business Report",
            data=report,
            file_name="business_analysis_report.txt",
            mime="text/plain"
        )