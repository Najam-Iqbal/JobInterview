from transformers import pipeline
import streamlit as st

# Load Models
question_model = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to generate an interview question
def generate_question(field="software development"):
    prompt = f"Generate an interview question for {field}."
    question = question_model(prompt, max_length=50, num_return_sequences=1, truncation=True)[0]['generated_text']
    return question

# Function to analyze answer sentiment
def analyze_answer(answer):
    result = sentiment_model(answer)
    sentiment = result[0]['label']
    confidence = result[0]['score']
    feedback = f"Your answer is {sentiment.lower()} with a confidence of {confidence:.2f}."
    return feedback

# Streamlit interface
def interview_app():
    st.title("Job Interview Preparation Assistant")
    st.write("Practice interview questions and receive AI-generated feedback!")

    # Create session state for the first question if it doesn't exist
    if "question" not in st.session_state:
        st.session_state.question = generate_question()

    # Display the interview question
    st.subheader("Interview Question")
    st.write(st.session_state.question)

    # Text area for user input (the answer)
    answer = st.text_area("Your Answer")

    # Submit button to analyze the answer
    if st.button("Submit Answer"):
        if answer.strip():  # Ensure that the user provides an answer
            feedback = analyze_answer(answer)
            st.write("Feedback:", feedback)
        else:
            st.warning("Please enter an answer before submitting.")

    # Next Question button
    if st.button("Next Question"):
        st.session_state.question = generate_question()  # Generate a new question
        st.write(st.session_state.question)

# Main function to run the app
if __name__ == "__main__":
    interview_app()
