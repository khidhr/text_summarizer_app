import streamlit as st
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration

st.set_page_config(
    page_title="Text Summarizer App",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #164863;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the smaller text summarization model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

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

        inputs = "summarize: " + user_input
        progress_bar = st.progress(0)
        summary = summarizer(inputs, max_length=150, min_length=40, do_sample=False)
        progress_bar.progress(100)
        st.subheader("Summary:")
        st.write(summary[0]['summary_text'])
    elif file:
        # Perform text summarization for uploaded file
        file_content = file.read().decode("utf-8")
        inputs = tokenizer.encode("summarize: " + file_content, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = summarizer.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_return_sequences=1)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text or upload a .txt file to summarize.")

st.markdown("### Model Information:")
st.markdown(f"This app uses the '{model_name}' model from the Hugging Face Transformers library for text summarization.")
st.markdown("For more information, see the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/t5.html).")

st.markdown("---")
st.subheader("Developed by [Halab Khidhr](https://github.com/khidhr)")
