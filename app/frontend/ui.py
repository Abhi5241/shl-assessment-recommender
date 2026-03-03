import streamlit as st
import requests

st.set_page_config(
    page_title="SHL Assessment Recommender",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/recommend"

st.title("🧠 SHL Assessment Recommendation System")
st.markdown(
    "AI-powered assessment recommendations using **Gemini LLM + Semantic Search**"
)

query = st.text_area(
    "Enter hiring requirement:",
    placeholder="Example: Python developer with teamwork and stakeholder skills",
    height=120
)

if st.button("🚀 Recommend Assessments"):

    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Analyzing requirements using AI..."):

            response = requests.post(
                API_URL,
                json={"query": query}
            )

            if response.status_code == 200:
                data = response.json()

                st.subheader("🧩 Detected Requirements")

                intent = data.get("detected_requirements", {})
                st.json(intent)

                st.subheader("✅ Recommended Assessments")

                results = data.get("recommended_assessments", [])
                
                st.subheader("🤖 Why These Assessments?")

                explanation = data.get("ai_explanation", "")

                st.markdown(explanation)

                if not results:
                    st.info("No recommendations found.")
                else:
                    for item in results:

                        with st.container(border=True):

                            col1, col2 = st.columns([4, 1])

                            with col1:
                                st.markdown(f"### {item['name']}")
                                st.write(item["description"])

                                st.write(
                                    f"⏱ Duration: {item['duration']} mins | "
                                    f"🧪 Type: {item['test_type']} | "
                                    f"🌐 Remote: {item['remote_support']}"
                                )

                            with col2:
                                st.link_button(
                                    "View",
                                    item["url"]
                                )

            else:
                st.error("API Error. Make sure FastAPI server is running.")