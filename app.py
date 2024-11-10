from transformers import pipeline
import streamlit as st

# Load Hugging Face models
question_model = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")
sentiment_model = pipeline("sentiment-analysis")

# Function to generate interview questions
def generate_question(field="software development"):
    prompt = f"Generate an interview question for {field}."
    question = question_model(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
    return question

# Function to analyze the answer sentiment
def analyze_answer(answer):
    result = sentiment_model(answer)
    sentiment = result[0]['label']
    confidence = result[0]['score']
    feedback = f"Your answer is {sentiment.lower()} with a confidence of {confidence:.2f}."
    return feedback

# Streamlit interface for the app
def interview_app():
    st.title("Job Interview Preparation Assistant")
    st.write("Practice interview questions and receive AI-generated feedback!")

    # Generate the first question
    if "question" not in st.session_state:
        st.session_state.question = generate_question()

    st.subheader("Interview Question")
    st.write(st.session_state.question)

    # Input for user's answer
    answer = st.text_area("Your Answer")

    # Button to submit the answer for sentiment analysis
    if st.button("Submit Answer"):
        if answer.strip():  # Check if the answer is not empty
            feedback = analyze_answer(answer)
            st.write("Feedback:", feedback)
        else:
            st.warning("Please enter an answer before submitting.")

    # Button to generate the next question
    if st.button("Next Question"):
        st.session_state.question = generate_question()
        st.write(st.session_state.question)

# Main function to run the app
if __name__ == "__main__":
    interview_app()
