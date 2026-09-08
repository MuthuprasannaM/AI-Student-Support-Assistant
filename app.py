import streamlit as st
from langchain_core.messages import HumanMessage
from main import app  # Import your compiled LangGraph workflow

st.set_page_config(page_title="Campus AI Assistant", page_icon="🎓")
st.title("🎓 AI Student Support Assistant")

# Maintain session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Session config for LangGraph memory
config = {"configurable": {"thread_id": "streamlit_user_session"}}

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about syllabus, regulations, or exams..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = app.invoke(
            {"messages": [HumanMessage(content=prompt)]}, 
            config=config
        )
        reply = response["messages"][-1].content
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})