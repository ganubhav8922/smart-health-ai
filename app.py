import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Load Custom Tools ---
tools = []
try:
    from tools import get_weather_precautions, emergency_hospital_locator_and_router
    tools = [get_weather_precautions, emergency_hospital_locator_and_router]
except ImportError:
    pass

# --- 2. Page Configuration ---
st.set_page_config(page_title="Smart Health Guide AI", page_icon="🏥", layout="wide")

st.title("AI Agent")
st.caption("Native Tool Binding Engine + Local RAG Knowledge + Live Weather Analysis")

# --- 3. Sidebar: Location Controls ---
st.sidebar.header("📍 Location Control Center")
lat = st.sidebar.number_input("Latitude", value=26.4499, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=80.3319, format="%.4f")

# --- 4. Initialize OpenAI LLM ---
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ `OPENAI_API_KEY` environment variable is not set. Please add it in Render settings.")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=api_key
)

# --- 5. System Prompt & Agent Setup ---
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        f"You are a helpful and accurate medical/health AI assistant. "
        f"The user's current location coordinates are Latitude: {lat}, Longitude: {lon}. "
        "Use your available tools when necessary to assist the user."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

if tools:
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
else:
    agent_executor = None

# --- 6. Session State for Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# --- 7. Chat Input & Processing ---
if user_input := st.chat_input("Describe symptoms or ask for health/emergency guidance..."):
    # Display user input
    with st.chat_message("user"):
        st.write(user_input)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                if agent_executor:
                    response = agent_executor.invoke({
                        "input": user_input,
                        "chat_history": st.session_state.chat_history
                    })
                    reply = response["output"]
                else:
                    response = llm.invoke(user_input)
                    reply = response.content

                st.write(reply)

                # Save interaction to history
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=reply))

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
