import streamlit as st
from streamlit_chat import message
from main import run_llm

st.header("Lexify Legal Helper")

prompt = st.text_input("Prompt", placeholder="Enter your question here...")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []


if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(query=prompt)
        formatted_response = generated_response['result']
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)

if st.session_state["chat_answers_history"]:
    for generated_answer, user_query in zip(st.session_state["chat_answers_history"],  st.session_state["user_prompt_history"]):
        message(user_query, is_user=True)
        message(generated_answer)
