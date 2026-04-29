import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Luxury Real Estate Predictor",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Custom Premium Styling
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.main {
    background-color: #0e1117;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: 700;
    color: #EAECEE;
    text-align: center;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #A6ACAF;
    margin-bottom: 30px;
}

/* Cards */
.card {
    background-color: #1c1f26;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}

/* Highlight */
.highlight {
    color: #F4D03F;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("real_estate_model.pkl")

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="title">🏠 Luxury Real Estate Valuation</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered property pricing for premium decision-making</div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Layout Split
# -----------------------------
left, right = st.columns([1, 2])

# -----------------------------
# LEFT PANEL (Inputs)
# -----------------------------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Property Details")

    area = st.number_input("Area (sqft)", 100, 10000, 1000)
    bhk = st.number_input("BHK", 1, 10, 2)

    locality = st.text_input("Locality", "Sector 56")
    builder = st.text_input("Builder Name", "DLF")

    property_type = st.selectbox("Property Type", ["Flat", "House"])
    status = st.selectbox("Status", ["Ready to move", "Under Construction"])
    rera = st.selectbox("RERA Approval", ["Yes", "No"])
    flat_type = st.selectbox("Flat Type", ["Builder Floor", "Apartment"])

    predict_btn = st.button("💎 Predict Price")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# RIGHT PANEL (Results)
# -----------------------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Valuation Insights")

    if predict_btn:
        input_data = {
            "area": area,
            "bhk_count": bhk,
            "locality": locality,
            "builder_name": builder,
            "property_type": property_type,
            "status": status,
            "rera_approval": rera,
            "flat_type": flat_type
        }

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]
        price_per_sqft = prediction / area

        col1, col2 = st.columns(2)

        col1.metric("💰 Estimated Value", f"₹ {round(prediction, 2):,}")
        col2.metric("📈 Price / Sqft", f"₹ {round(price_per_sqft, 2):,}")

        st.markdown("---")

        # Premium Insight Box
        if price_per_sqft > 10000:
            st.markdown("### 🏆 Premium Property")
            st.markdown("This property falls in the **high-value luxury segment**.")
        elif price_per_sqft < 5000:
            st.markdown("### 💰 Budget Opportunity")
            st.markdown("This property is **value-driven and affordable**.")
        else:
            st.markdown("### ⚖️ Mid-Segment Asset")
            st.markdown("This property is **moderately priced with balanced value**.")

    else:
        st.info("Enter property details and click **Predict Price**")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
---
<p style='text-align: center; color: gray;'>
Premium Real Estate Analytics • Built with ML
</p>
""", unsafe_allow_html=True)