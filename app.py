import streamlit as st
import asyncio
import os
from src.reasoning import ReasoningAgent

# Page Config
st.set_page_config(
    page_title="MedGraph-RAG",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS for "Neo-Brutalist" feel (light touch)
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 0px !important;
        border: 2px solid #000 !important;
        box-shadow: 4px 4px 0px #000 !important;
        margin-bottom: 1rem !important;
    }
    .stTextInput > div > div > input {
        border: 2px solid #000 !important;
        border-radius: 0px !important;
        box-shadow: 2px 2px 0px #000 !important;
    }
    h1 {
        text-transform: uppercase;
        font-weight: 900 !important;
        letter-spacing: -2px;
    }
</style>
""", unsafe_allow_html=True)

# Application Title
st.title("🧬 MedGraph-RAG")
st.caption("Hybrid Retrieval & Chain-of-Graph Reasoning System")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    model_name = st.selectbox("Model", ["gemini-2.0-flash", "gemini-2.0-pro (Coming Soon)"])
    st.divider()
    
    st.info("System Status: **Active**")
    st.text(f"Backend: {model_name}")
    
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Reasoning Agent Wrapper
async def run_agent(query: str):
    agent = ReasoningAgent()
    
    # We want to capture the intermediate steps (Reasoning Traces)
    # The current agent.run() returns a final dict. 
    # To stream steps in a real app, we'd need to use agent.graph.astream()
    # For this prototype, we'll run it and display the traces from the result dict.
    
    return await agent.run(query)

# Input
if prompt := st.chat_input("Ask a medical question..."):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.status("🧠 Thinking (Reasoning Trace)...", expanded=True) as status:
            st.write("Initializing Agent...")
            
            # Run the agent (Async wrapper for streamlit)
            try:
                result = asyncio.run(run_agent(prompt))
                
                # Show key steps from the result
                if "plan" in result:
                    st.subheader("Plan")
                    for step in result["plan"]:
                        st.markdown(f"- {step}")
                
                # We could ideally capture the retrieving steps too if we modify the agent to stream
                # For now, we show the final reflection
                if "reflection" in result:
                    st.divider()
                    st.markdown(f"**Reflection:** {result['reflection']}")
                    
                status.update(label="✅ Reasoning Complete", state="complete", expanded=False)
                
                # Final Answer
                answer = result.get("answer", "No answer generated.")
                st.markdown(answer)
                
                # Display Citations
                if "context" in result and result["context"]:
                    st.divider()
                    st.subheader("📚 Sources")
                    for idx, doc in enumerate(result["context"]):
                        # Deduplicate or just show top relevant?
                        # For now, show unique ones based on content presence
                        with st.expander(f"Source {idx+1}: {doc.get('metadata', {}).get('doc_title', 'Unknown')}"):
                            st.caption(f"Section: {doc.get('metadata', {}).get('section_title', 'Unknown')}")
                            st.text(doc.get('content'))
                
                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                status.update(label="❌ Failed", state="error")
