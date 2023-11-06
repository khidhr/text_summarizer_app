import streamlit as st
from transformers import pipeline
st.set_page_config(
    page_title="Text Summarizer App",
    page_icon=":pencil:",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #3D30A2;
    }
    </style>
    """, unsafe_allow_html=True)
# Load the text summarization model from Hugging Face
summarizer = pipeline("summarization")

# Streamlit UI components
st.title("Text Summarizer App")
st.markdown("### Instructions:")
st.markdown("- Enter text in the text area to summarize.")
st.markdown("- Alternatively, upload a .txt file for summarization.")
user_input = st.text_area("Enter Text:")
file = st.file_uploader("Upload a Text File (.txt)", type=["txt"])

if st.button("Summarize"):
    if user_input:
        # Perform text summarization for user 
        progress_bar = st.progress(0)

        summary = summarizer(user_input, max_length=150, do_sample=False)
        progress_bar.progress(100)

        st.subheader("Summary:")
        st.write(summary[0]["summary_text"])
    elif file:
        # Perform text summarization for uploaded file
        file_content = file.read().decode("utf-8")
        summary = summarizer(file_content, max_length=150, do_sample=False)
        st.subheader("Summary:")
        st.write(summary[0]["summary_text"])
    else:
        st.warning("Please enter some text or upload a .txt file to summarize.")

st.markdown("### Model Information:")
st.markdown("This app uses the 't5-base' model from the Hugging Face Transformers library for text summarization.")
st.markdown("The 't5-base' model is a text-to-text transformer trained on a mixture of unsupervised and supervised tasks.")
st.markdown("For more information, see the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/t5.html).")

st.markdown("---")
st.markdown("Developed by [Halab Khidhr](https://github.com/khidhr)")
