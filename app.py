import streamlit as st
import sqlite3
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================
# 0. SECURITY & API KEY CONTROL
# ==========================================
import os
MY_GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
# Import tools
from tools import get_weather_precautions, find_emergency_hospital

# Helper function to unpack Gemini's block output cleanly
def parse_ai_content(content) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
        return "\n".join(text_parts) if text_parts else ""
    return str(content)

# ==========================================
# 1. PERMANENT DATABASE MEMORY (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect("health_chat_history.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_message_to_db(role: str, content: str):
    conn = sqlite3.connect("health_chat_history.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_logs (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

def load_chat_history_from_db():
    conn = sqlite3.connect("health_chat_history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM chat_logs ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]

init_db()

# ==========================================
# 2. VECTOR KNOWLEDGE BASE (RAG Function)
# ==========================================
def search_medical_knowledge(query: str) -> str:
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory="./chroma_db_storage", embedding_function=embedding_model)
        docs = db.similarity_search(query, k=2)
        if docs:
            return "\n\n".join([d.page_content for d in docs])
        return "No local medical guidelines found for this query."
    except Exception as e:
        return f"RAG error: {str(e)}"

# ==========================================
# 3. STREAMLIT FRONTEND WEB UI
# ==========================================
st.set_page_config(page_title="Smart Health Guide AI", page_icon="🏥", layout="wide")
st.title("🏥 Smart Health Guide AI Agent")
st.caption("Native Tool Binding Engine + Local RAG Knowledge + Live Weather Analysis")

st.sidebar.header("📍 Location Control Center")
user_lat = st.sidebar.number_input("Latitude", value=26.4499, format="%.4f")
user_lon = st.sidebar.number_input("Longitude", value=80.3319, format="%.4f")

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history_from_db()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ==========================================
# 4. REASONING & EXECUTION LOOP
# ==========================================
if user_input := st.chat_input("Describe symptoms or ask for health/emergency guidance..."):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    save_message_to_db("user", user_input)

    @tool
    def medical_knowledge_base_lookup(query: str) -> str:
        """Look up medical guidelines and symptoms in local records."""
        return search_medical_knowledge(query)

    @tool
    def live_weather_condition_evaluator() -> str:
        """Fetch weather precautions based on coordinates."""
        return get_weather_precautions(user_lat, user_lon)

    @tool
    def emergency_hospital_locator_and_router() -> str:
        """Find the nearest emergency hospital."""
        return find_emergency_hospital(user_lat, user_lon)

    ai_tools = [
        medical_knowledge_base_lookup, 
        live_weather_condition_evaluator, 
        emergency_hospital_locator_and_router
    ]

    try:
        # THE FIX: Updated strictly to gemini-3.5-flash
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=MY_GEMINI_KEY, 
            temperature=0.1
        )
        llm_with_tools = llm.bind_tools(ai_tools)
        
        contextual_prompt = (
            f"User Request: {user_input}\n"
            f"User Location: Lat {user_lat}, Lon {user_lon}."
        )
        
        run_messages = [
            SystemMessage(content="You are an advanced medical assistant. Give helpful advice and use tools when relevant."),
            HumanMessage(content=contextual_prompt)
        ]

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                ai_msg = llm_with_tools.invoke(run_messages)
                
                if ai_msg.tool_calls:
                    run_messages.append(ai_msg)
                    for tool_call in ai_msg.tool_calls:
                        t_name = tool_call["name"]
                        t_args = tool_call["args"]
                        matched_tool = next(t for t in ai_tools if t.name == t_name)
                        tool_output = matched_tool.invoke(t_args)
                        run_messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
                    
                    final_msg = llm_with_tools.invoke(run_messages)
                    ai_response = parse_ai_content(final_msg.content)
                else:
                    ai_response = parse_ai_content(ai_msg.content)
                
                if not ai_response:
                    ai_response = "I evaluated your input, but no additional recommendations were generated."

                st.write(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                save_message_to_db("assistant", str(ai_response))
                
    except Exception as error:
        st.error(f"Execution Error: {str(error)}")