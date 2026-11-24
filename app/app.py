import streamlit as st
from src.predict import clean_review, predict_review


# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="ReviewTrellis",
    page_icon="🌿",
    layout="wide"
)


# =====================================
# CUSTOM DESIGN
# =====================================

st.markdown("""
<style>

.stApp {
    background-color: #F7F8FA;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #0B2A4A;
}

.subtitle {
    font-size: 16px;
    color: #64748B;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: white;
    border: 1px solid #E2E8F0;
}

.metric-box {
    padding: 18px;
    background-color: white;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    text-align: center;
}

.metric-title {
    color: #64748B;
    font-size: 14px;
}

.metric-value {
    color: #0B2A4A;
    font-size: 26px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.title("🌿 ReviewTrellis")

    st.write("Fake Product Review Detection")

    st.divider()

    st.markdown("### About the Project")

    st.write(
        "ReviewTrellis uses Machine Learning "
        "to identify review patterns associated "
        "with genuine and computer-generated reviews."
    )

    st.divider()

    st.markdown("### Best ML Model")

    st.info("Linear Support Vector Machine")

    st.markdown("### Model Accuracy")

    st.success("94.55%")

    st.caption(
        "Accuracy measured on the test dataset."
    )


# =====================================
# MAIN DASHBOARD HEADER
# =====================================

st.markdown(
    '<div class="main-title">Review Analysis Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze product reviews using your trained ML model.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# =====================================
# REVIEW INPUT
# =====================================

st.markdown("### Enter Product Review")

sample_review = (
    "This product is amazing and works perfectly. "
    "The quality is excellent and I am very happy "
    "with my purchase."
)

if st.button("Use Sample Review"):

    st.session_state["review_input"] = sample_review


review = st.text_area(
    "Paste the review here:",
    height=180,
    placeholder="Enter a product review...",
    key="review_input"
)


# =====================================
# REVIEW STATISTICS
# =====================================

word_count = len(review.split())

character_count = len(review)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Word Count</div>
            <div class="metric-value">{word_count}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">Character Count</div>
            <div class="metric-value">{character_count}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        """
        <div class="metric-box">
            <div class="metric-title">ML Model</div>
            <div class="metric-value">Linear SVM</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# =====================================
# ANALYZE REVIEW
# =====================================

if st.button("🔍 Analyze Review", use_container_width=True):

    if review.strip() == "":

        st.warning("Please enter a product review.")

    elif clean_review(review) == "":

        st.warning("Please enter valid review text.")

    else:

        prediction = predict_review(review)

        st.session_state["last_prediction"] = prediction

        if "history" not in st.session_state:

            st.session_state["history"] = []

        result_text = (
            "Likely Genuine"
            if prediction == 0
            else "Potentially Fake"
        )

        st.session_state["history"].append({
            "review": review,
            "result": result_text
        })


# =====================================
# DISPLAY RESULT
# =====================================

if "last_prediction" in st.session_state:

    prediction = st.session_state["last_prediction"]

    st.divider()

    st.markdown("### Analysis Result")

    if prediction == 0:

        st.success("✅ Likely Genuine Review")

        st.write(
            "The model classified this review as likely genuine "
            "based on patterns learned from the training dataset."
        )

    else:

        st.error("⚠️ Potentially Fake Review")

        st.write(
            "The model identified patterns associated with "
            "computer-generated reviews in the training dataset."
        )

    st.info(
        "This result is an ML prediction, not proof "
        "that the review is genuine or fake."
    )


# =====================================
# RECENT ANALYSIS HISTORY
# =====================================

if "history" in st.session_state:

    st.divider()

    st.markdown("### Recent Review Analysis")

    for item in reversed(st.session_state["history"][-5:]):

        with st.expander(item["result"]):

            st.write(item["review"])

    if st.button("Clear Analysis History"):

        st.session_state["history"] = []

        st.rerun()


# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "ReviewTrellis | Machine Learning Mini Project | "
    "Fake Product Review Detection"
)