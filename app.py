import streamlit as st
from backend.rag_engine import RAGEngine

st.set_page_config(page_title="AI Emergency First Aid Guide")

st.title("🚑 AI Emergency First Aid Guide")
st.write("Describe the emergency and get step-by-step first aid help.")

st.warning("⚠ For educational purposes only. Call emergency services in real emergencies.")

rag = RAGEngine()

st.subheader("📝 Enter Emergency Situation:")
query = st.text_area("Type here...", height=150)

if st.button("Get First Aid Help"):
    if query.strip() == "":
        st.error("Please enter an emergency.")
    else:
        with st.spinner("Generating guidance..."):
            result = rag.get_response(query)

        st.subheader("✅ First Aid Instructions:")
        st.success(result)
