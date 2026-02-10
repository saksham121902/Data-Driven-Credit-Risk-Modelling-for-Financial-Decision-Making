import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---- Page Config ----
st.set_page_config(page_title="Credit Risk Assessment", page_icon="🏦", layout="wide")

# ---- Load Model ----
@st.cache_resource
def load_model():
    return joblib.load("credit_risk_model.pkl")

pipeline = load_model()

# ---- Helper ----
def risk_bucket(pd_val):
    if pd_val < 0.10:
        return "Low", "🟢"
    elif pd_val < 0.25:
        return "Medium", "🟡"
    else:
        return "High", "🔴"

# ---- Sidebar Inputs ----
st.sidebar.header("Applicant Details")

person_age = st.sidebar.slider("Age", 18, 80, 30)
person_income = st.sidebar.number_input("Annual Income (₹)", 5000, 500000, 50000, step=1000)
person_home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_emp_length = st.sidebar.slider("Employment Length (years)", 0, 40, 5)
loan_intent = st.sidebar.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
loan_grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.sidebar.number_input("Loan Amount (₹)", 500, 50000, 10000, step=500)
loan_int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 25.0, 12.0, step=0.1)
loan_percent_income = st.sidebar.slider("Loan-to-Income Ratio", 0.0, 1.0, 0.2, step=0.01)
cb_person_default_on_file = st.sidebar.selectbox("Previous Default on File", ["N", "Y"])
cb_person_cred_hist_length = st.sidebar.slider("Credit History Length (years)", 2, 30, 5)

# ---- Build Input DataFrame ----
input_data = pd.DataFrame([{
    "person_age": person_age,
    "person_income": person_income,
    "person_home_ownership": person_home_ownership,
    "person_emp_length": float(person_emp_length),
    "loan_intent": loan_intent,
    "loan_grade": loan_grade,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_default_on_file": cb_person_default_on_file,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
}])

# ---- Main Content ----
st.title("🏦 Credit Risk Assessment System")
st.markdown("Predict the **Probability of Default** and risk category for a loan applicant.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Applicant Summary")
    display_df = input_data.T.rename(columns={0: "Value"})
    display_df.index = display_df.index.map({
        "person_age": "Age",
        "person_income": "Annual Income (₹)",
        "person_home_ownership": "Home Ownership",
        "person_emp_length": "Employment Length (years)",
        "loan_intent": "Loan Purpose",
        "loan_grade": "Loan Grade",
        "loan_amnt": "Loan Amount (₹)",
        "loan_int_rate": "Interest Rate (%)",
        "loan_percent_income": "Loan-to-Income Ratio",
        "cb_person_default_on_file": "Previous Default on File",
        "cb_person_cred_hist_length": "Credit History Length (years)",
    })
    display_df["Value"] = display_df["Value"].astype(str)
    st.table(display_df)

# ---- Prediction ----
if st.sidebar.button("Assess Risk", use_container_width=True):
    pd_val = pipeline.predict_proba(input_data)[0][1]
    risk, emoji = risk_bucket(pd_val)

    with col2:
        st.subheader("Risk Assessment Result")
        st.metric("Probability of Default", f"{pd_val:.1%}")
        st.markdown(f"### Risk Level: {emoji} **{risk}**")

        if risk == "Low":
            st.success("This applicant has a low risk of default. Likely eligible for favorable terms.")
        elif risk == "Medium":
            st.warning("Moderate risk. Consider additional verification or adjusted terms.")
        else:
            st.error("High risk of default. Exercise caution — may require collateral or higher rates.")

    # ---- How Could This Applicant Reduce Risk? ----
    st.subheader("How Could This Applicant Reduce Risk?")

    try:
        # Get transformed features and feature names
        X_transformed = pipeline.named_steps["preprocess"].transform(input_data)
        rf_model = pipeline.named_steps["model"]
        feature_names = list(pipeline.named_steps["preprocess"].get_feature_names_out())

        if hasattr(X_transformed, "toarray"):
            data_row = X_transformed.toarray()[0]
        else:
            data_row = np.array(X_transformed)[0]

        # Compute per-feature contribution to PD using tree traversal
        # Each tree votes; we measure how much each feature pushes PD up
        importances = rf_model.feature_importances_
        feature_contributions = np.abs(data_row) * importances
        total = feature_contributions.sum()
        if total > 0:
            feature_pcts = (feature_contributions / total) * pd_val * 100
        else:
            feature_pcts = np.zeros_like(feature_contributions)

        # Build a mapping of feature names to human-readable suggestions
        suggestion_map = {
            "person_age": "Younger applicants tend to carry higher risk; age is not changeable but longer credit history helps.",
            "person_income": "A higher income reduces default probability. Consider ways to increase earnings.",
            "person_emp_length": "Longer employment tenure signals income stability to lenders.",
            "loan_amnt": "Requesting a smaller loan amount would reduce risk exposure.",
            "loan_int_rate": "A lower interest rate reduces repayment burden. Improving creditworthiness can help.",
            "loan_percent_income": "Lowering the loan-to-income ratio by borrowing less or earning more reduces risk.",
            "cb_person_cred_hist_length": "Building a longer credit history strengthens the applicant's profile.",
            "person_home_ownership_RENT": "Stable housing or home ownership may slightly reduce risk.",
            "person_home_ownership_OWN": "Owning a home is generally positive for creditworthiness.",
            "person_home_ownership_MORTGAGE": "Having a mortgage indicates financial commitment and stability.",
            "person_home_ownership_OTHER": "Stable housing or home ownership may slightly reduce risk.",
            "loan_grade_A": "This is already the best loan grade.",
            "loan_grade_B": "Improving this factor could reduce default probability.",
            "loan_grade_C": "Improving this factor could reduce default probability.",
            "loan_grade_D": "A better loan grade (A–C) would significantly lower risk.",
            "loan_grade_E": "A better loan grade (A–C) would significantly lower risk.",
            "loan_grade_F": "A better loan grade (A–C) would significantly lower risk.",
            "loan_grade_G": "A better loan grade (A–C) would significantly lower risk.",
            "loan_intent_EDUCATION": "Education loans are generally viewed as investments in future earning potential.",
            "loan_intent_MEDICAL": "Medical loans are essential; maintaining insurance coverage can reduce need.",
            "loan_intent_VENTURE": "Improving this factor could reduce default probability.",
            "loan_intent_PERSONAL": "Improving this factor could reduce default probability.",
            "loan_intent_DEBTCONSOLIDATION": "Consolidating debt can be positive if it lowers overall payments.",
            "loan_intent_HOMEIMPROVEMENT": "Home improvement loans can add value to owned property.",
            "cb_person_default_on_file_Y": "Having a previous default significantly raises risk. Clearing obligations helps.",
            "cb_person_default_on_file_N": "No previous default is a positive signal.",
        }

        # Sort by impact and show top contributors
        sorted_idx = np.argsort(feature_pcts)[::-1]
        top_n = 5
        shown = 0

        for idx in sorted_idx:
            if shown >= top_n:
                break
            fname = feature_names[idx]
            impact = feature_pcts[idx]
            if impact < 0.1:
                continue

            # Clean feature name for display (remove num__/cat__ prefix)
            display_name = fname
            # Strip pipeline prefixes like num__ or cat__
            for prefix in ("num__", "cat__"):
                if display_name.startswith(prefix):
                    display_name = display_name[len(prefix):]
            # Convert underscores to readable labels
            readable_map = {
                "person_age": "Age",
                "person_income": "Annual Income",
                "person_emp_length": "Employment Length",
                "person_home_ownership_RENT": "Home Ownership: Rent",
                "person_home_ownership_OWN": "Home Ownership: Own",
                "person_home_ownership_MORTGAGE": "Home Ownership: Mortgage",
                "person_home_ownership_OTHER": "Home Ownership: Other",
                "loan_intent_EDUCATION": "Loan Purpose: Education",
                "loan_intent_MEDICAL": "Loan Purpose: Medical",
                "loan_intent_VENTURE": "Loan Purpose: Venture",
                "loan_intent_PERSONAL": "Loan Purpose: Personal",
                "loan_intent_DEBTCONSOLIDATION": "Loan Purpose: Debt Consolidation",
                "loan_intent_HOMEIMPROVEMENT": "Loan Purpose: Home Improvement",
                "loan_grade_A": "Loan Grade: A",
                "loan_grade_B": "Loan Grade: B",
                "loan_grade_C": "Loan Grade: C",
                "loan_grade_D": "Loan Grade: D",
                "loan_grade_E": "Loan Grade: E",
                "loan_grade_F": "Loan Grade: F",
                "loan_grade_G": "Loan Grade: G",
                "loan_amnt": "Loan Amount",
                "loan_int_rate": "Interest Rate",
                "loan_percent_income": "Loan-to-Income Ratio",
                "cb_person_default_on_file_Y": "Previous Default: Yes",
                "cb_person_default_on_file_N": "Previous Default: No",
                "cb_person_cred_hist_length": "Credit History Length",
            }
            display_name = readable_map.get(display_name, display_name.replace("_", " ").title())
            # Find suggestion
            suggestion = None
            for key, sug in suggestion_map.items():
                if key in fname:
                    suggestion = sug
                    break
            if suggestion is None:
                suggestion = "Improving this factor could reduce default probability."

            st.markdown(f"• `{display_name}`")
            st.markdown(f"&nbsp;&nbsp;Impact on PD: **{impact:.1f}%**")
            st.markdown(f"&nbsp;&nbsp; *Suggestion:* {suggestion}")
            st.markdown("")
            shown += 1

        if shown == 0:
            st.success("✅ This applicant's profile already looks strong! No major risk-reduction steps needed.")

    except Exception as e:
        st.info(f"Risk reduction analysis unavailable: {e}")

else:
    with col2:
        st.info("👈 Fill in the applicant details and click **Assess Risk** to get a prediction.")
