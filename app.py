# import streamlit as st
# import joblib
# import pandas as pd
# import os

# # -------------------------
# # PAGE CONFIG (MUST BE FIRST)
# # -------------------------
# st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

# st.title("🎓 Student Dropout Prediction System")
# st.write("Enter student details below to predict dropout risk.")

# # -------------------------
# # LOAD MODEL
# # -------------------------
# @st.cache_resource
# def load_model():
#     model_path = "best_xgb.pkl"
#     st.write("Loading model from:", os.path.abspath(model_path))
#     model = joblib.load(model_path)
#     return model

# try:
#     model = load_model()
#     st.success("Model loaded successfully ✔")
# except Exception as e:
#     st.error(f"Model loading failed: {e}")
#     st.stop()

# # -------------------------
# # INPUT OPTIONS
# # -------------------------
# COUNTIES = [
#     'KAKAMEGA', 'KAJIADO', 'NAIROBI', 'KIAMBU', 'KITUI', 'MACHAKOS',
#     'UASIN GISHU', 'MOMBASA', 'MIGORI', 'ELGEYO MARAKWET', 'BUNGOMA',
#     'NYAMIRA', 'KILIFI', 'BOMET', 'TRANS NZOIA', 'KISUMU', 'HOMA BAY',
#     'MERU', 'THARAKA NITHI', 'KWALE', "MURANG'A", 'NYERI', 'BUSIA',
#     'NAKURU', 'KISII', 'KIRINYAGA', 'NANDI', 'VIHIGA', 'SIAYA',
#     'MAKUENI', 'NYANDARUA', 'TAITA TAVETA', 'LAIKIPIA', 'KERICHO',
#     'LAMU', 'NAROK', 'SAMBURU', 'BARINGO', 'EMBU', 'WEST POKOT',
#     'GARISSA', 'MARSABIT', 'TURKANA', 'TANA RIVER', 'ISIOLO',
#     'MANDERA', 'WAJIR'
# ]

# # -------------------------
# # UI LAYOUT
# # -------------------------
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Personal Information")

#     EnrollmentAge = st.number_input("Age", 15, 60, 20)
#     gender = st.selectbox("Gender", ["Male", "Female"])
#     county = st.selectbox("County", sorted(COUNTIES))
#     scholarship = st.selectbox("Scholarship Applied", [0, 1],
#                                format_func=lambda x: "Yes" if x == 1 else "No")

# with col2:
#     st.subheader("Academic Information")

#     program_cost = st.number_input("Program Cost", 0.0, 1_000_000.0, 150000.0)
#     loan_allocated = st.number_input("Total Loan Allocated", 0.0, 1_000_000.0, 40000.0)

#     sponsored = st.selectbox("Sponsored", ["GovtSponsored", "SelfSponsored"])
#     loan_status = st.selectbox("Loan Status", ["Partially Disbursed", "Allocated", "Cancelled", "Deffered"])
#     course_cat = st.selectbox(
#         "Course Category",
#         ["EngineeringTechnology", "Humanity", "ICT", "Education", "Medical", "ScienceAgriculture"]
#     )

# # -------------------------
# # BUILD INPUT DATAFRAME (CRITICAL FIX)
# # -------------------------
# input_df = pd.DataFrame([{
#     'Gender': gender,
#     'County': county,
#     'CourseCategory': course_cat,
#     'LoanStatus': loan_status,
#     'Sponsored': sponsored,
#     'ProgramCost': program_cost,
#     'TotalLoanAllocated': loan_allocated,
#     'ScholarshipApplied': scholarship,
#     'EnrollmentAge': EnrollmentAge
# }])

# # -------------------------
# # PREDICTION
# # -------------------------
# if st.button("Predict Dropout Risk"):

#     try:
#         # Ensure correct type
#         if not isinstance(input_df, pd.DataFrame):
#             raise ValueError("Input is not a valid DataFrame")

#         prediction = model.predict(input_df)[0]
#         probability = model.predict_proba(input_df)[0][1]

#         st.markdown("---")

#         if prediction == 1:
#             st.error("⚠️ HIGH RISK: Student likely to drop out")
#             st.write(f"Dropout Probability: **{probability:.2%}**")
#         else:
#             st.success("✅ LOW RISK: Student likely to continue studies")
#             st.write(f"Retention Probability: **{1 - probability:.2%}**")

#     except Exception as e:
#         st.error("Prediction failed")
#         st.exception(e)

# # -------------------------
# # DEBUG SECTION
# # -------------------------
# with st.expander("Show Technical Input Data"):
#     st.dataframe(input_df)



import streamlit as st
import joblib
import pandas as pd
import os

# -------------------------
# PAGE CONFIG (MUST BE FIRST)
# -------------------------
st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

# -------------------------
# STYLING
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Background ── */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1523050854058-8df90110c9f1?w=1920&q=90");
    background-size: cover;
    background-position: center top;
    background-attachment: fixed;
}

/* Deep overlay */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background: linear-gradient(
        160deg,
        rgba(2, 18, 38, 0.91) 0%,
        rgba(4, 40, 70, 0.87) 40%,
        rgba(10, 20, 50, 0.93) 100%
    );
    z-index: 0;
    pointer-events: none;
}

/* ── Layout ── */
.main .block-container {
    position: relative;
    z-index: 1;
    padding-top: 2.5rem;
    max-width: 1200px;
}

/* ── Title ── */
h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 3.2rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    text-shadow: 0 4px 30px rgba(0, 180, 220, 0.35);
    letter-spacing: 0.5px;
    line-height: 1.15 !important;
}

/* ── Subheaders ── */
h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.45rem !important;
    font-weight: 600 !important;
    color: #4dd9f5 !important;
    border-bottom: 1.5px solid rgba(77, 217, 245, 0.25);
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem !important;
    letter-spacing: 0.3px;
}

/* ── Body text ── */
p, label, div[data-testid="stText"],
.stMarkdown p {
    font-family: 'DM Sans', sans-serif !important;
    color: #b8cfe8 !important;
    font-size: 0.95rem !important;
}

/* ── Column cards ── */
div[data-testid="column"] > div {
    background: rgba(6, 30, 60, 0.55);
    border: 1px solid rgba(77, 217, 245, 0.18);
    border-radius: 20px;
    padding: 2rem 2rem;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    box-shadow:
        0 0 0 1px rgba(77, 217, 245, 0.06),
        0 12px 40px rgba(0, 0, 0, 0.45),
        inset 0 1px 0 rgba(255,255,255,0.06);
}

/* ── Number inputs ── */
.stNumberInput input {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(77, 217, 245, 0.25) !important;
    border-radius: 10px !important;
    color: #e8f4ff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
.stNumberInput input:focus {
    border-color: rgba(77, 217, 245, 0.7) !important;
    box-shadow: 0 0 0 2px rgba(77, 217, 245, 0.15) !important;
}

/* ── Selectboxes ── */
.stSelectbox > div > div,
div[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(77, 217, 245, 0.25) !important;
    border-radius: 10px !important;
    color: #e8f4ff !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stSelectbox div[data-baseweb="select"] span,
div[data-baseweb="select"] * {
    color: #e8f4ff !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Labels ── */
div[data-testid="stWidgetLabel"] p,
.stSelectbox label,
.stNumberInput label {
    color: #7ec8e3 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Predict Button ── */
.stButton > button {
    background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%) !important;
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.7rem 2.8rem !important;
    border: none !important;
    border-radius: 50px !important;
    letter-spacing: 0.6px;
    box-shadow: 0 4px 24px rgba(0, 114, 255, 0.45) !important;
    transition: all 0.25s ease !important;
    margin-top: 0.8rem;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 35px rgba(0, 114, 255, 0.65) !important;
    background: linear-gradient(135deg, #33d6ff 0%, #1a8fff 100%) !important;
}

/* ── HIGH RISK box (red) ── */
.risk-high {
    background: linear-gradient(135deg, rgba(220, 30, 30, 0.18), rgba(180, 0, 0, 0.25));
    border: 1.5px solid rgba(255, 80, 80, 0.6);
    border-left: 5px solid #ff3333;
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    margin-top: 1.2rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 30px rgba(255, 50, 50, 0.2);
}
.risk-high .risk-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: #ff5555;
    margin-bottom: 0.4rem;
}
.risk-high .risk-stat {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    color: #ffaaaa;
    font-weight: 500;
}
.risk-high .risk-stat span {
    color: #ff3333;
    font-weight: 700;
    font-size: 1.4rem;
}

/* ── LOW RISK box (green) ── */
.risk-low {
    background: linear-gradient(135deg, rgba(0, 180, 100, 0.15), rgba(0, 140, 70, 0.22));
    border: 1.5px solid rgba(0, 220, 120, 0.5);
    border-left: 5px solid #00dd88;
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    margin-top: 1.2rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 30px rgba(0, 200, 100, 0.18);
}
.risk-low .risk-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: #00dd88;
    margin-bottom: 0.4rem;
}
.risk-low .risk-stat {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    color: #88ffcc;
    font-weight: 500;
}
.risk-low .risk-stat span {
    color: #00ff99;
    font-weight: 700;
    font-size: 1.4rem;
}

/* ── Divider ── */
hr {
    border-color: rgba(77, 217, 245, 0.15) !important;
    margin: 1.5rem 0 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(6, 30, 60, 0.4) !important;
    border: 1px solid rgba(77, 217, 245, 0.15) !important;
    border-radius: 10px !important;
    color: #7ec8e3 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── intro text ── */
.intro-text {
    font-family: 'DM Sans', sans-serif;
    color: #7ec8e3;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.title("🎓 Student Dropout Prediction System")
st.markdown('<p class="intro-text">Enter student details below to predict dropout risk.</p>', unsafe_allow_html=True)

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "best_xgb.pkl")
    model = joblib.load(model_path)
    return model

try:
    model = load_model()
    st.success("Model loaded successfully ✔")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# -------------------------
# INPUT OPTIONS
# -------------------------
COUNTIES = [
    'KAKAMEGA', 'KAJIADO', 'NAIROBI', 'KIAMBU', 'KITUI', 'MACHAKOS',
    'UASIN GISHU', 'MOMBASA', 'MIGORI', 'ELGEYO MARAKWET', 'BUNGOMA',
    'NYAMIRA', 'KILIFI', 'BOMET', 'TRANS NZOIA', 'KISUMU', 'HOMA BAY',
    'MERU', 'THARAKA NITHI', 'KWALE', "MURANG'A", 'NYERI', 'BUSIA',
    'NAKURU', 'KISII', 'KIRINYAGA', 'NANDI', 'VIHIGA', 'SIAYA',
    'MAKUENI', 'NYANDARUA', 'TAITA TAVETA', 'LAIKIPIA', 'KERICHO',
    'LAMU', 'NAROK', 'SAMBURU', 'BARINGO', 'EMBU', 'WEST POKOT',
    'GARISSA', 'MARSABIT', 'TURKANA', 'TANA RIVER', 'ISIOLO',
    'MANDERA', 'WAJIR'
]

# -------------------------
# UI LAYOUT
# -------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Personal Information")
    EnrollmentAge = st.number_input("Age", 15, 60, 20)
    gender = st.selectbox("Gender", ["Male", "Female"])
    county = st.selectbox("County", sorted(COUNTIES))
    scholarship = st.selectbox("Scholarship Applied", [0, 1],
                               format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    st.subheader("Academic Information")
    program_cost = st.number_input("Program Cost", 0.0, 1_000_000.0, 150000.0)
    loan_allocated = st.number_input("Total Loan Allocated", 0.0, 1_000_000.0, 40000.0)
    sponsored = st.selectbox("Sponsored", ["GovtSponsored", "SelfSponsored"])
    loan_status = st.selectbox("Loan Status", ["Partially Disbursed", "Allocated", "Cancelled", "Deffered"])
    course_cat = st.selectbox(
        "Course Category",
        ["EngineeringTechnology", "Humanity", "ICT", "Education", "Medical", "ScienceAgriculture"]
    )

# -------------------------
# BUILD INPUT DATAFRAME
# -------------------------
input_df = pd.DataFrame([{
    'Gender': gender,
    'County': county,
    'CourseCategory': course_cat,
    'LoanStatus': loan_status,
    'Sponsored': sponsored,
    'ProgramCost': program_cost,
    'TotalLoanAllocated': loan_allocated,
    'ScholarshipApplied': scholarship,
    'EnrollmentAge': EnrollmentAge
}])

# -------------------------
# PREDICTION
# -------------------------
if st.button("🔍 Predict Dropout Risk"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("---")

        if prediction == 1:
            dropout_pct = f"{probability:.2%}"
            st.markdown(f"""
            <div class="risk-high">
                <div class="risk-title">⚠️ HIGH RISK: Student likely to drop out</div>
                <div class="risk-stat">Dropout Probability: <span>{dropout_pct}</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            retention_pct = f"{(1 - probability):.2%}"
            st.markdown(f"""
            <div class="risk-low">
                <div class="risk-title">✅ LOW RISK: Student likely to continue studies</div>
                <div class="risk-stat">Retention Probability: <span>{retention_pct}</span></div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)

# -------------------------
# DEBUG SECTION
# -------------------------
with st.expander("Show Technical Input Data"):
    st.dataframe(input_df)