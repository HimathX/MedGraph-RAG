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
                
                # Display execution events in structured sections
                execution_events = result.get("execution_events", [])
                
                # Group events by type
                plan_events = [e for e in execution_events if e.get("type") == "plan_created"]
                tool_events = [e for e in execution_events if e.get("type") == "tool_call"]
                reflection_events = [e for e in execution_events if e.get("type") == "reflection"]
                answer_events = [e for e in execution_events if e.get("type") == "final_answer"]
                
                # 1. Planning Section
                if plan_events:
                    with st.expander("📋 Planning", expanded=True):
                        for event in plan_events:
                            st.markdown("**Generated Plan:**")
                            for i, step in enumerate(event.get("plan", []), 1):
                                st.markdown(f"{i}. {step}")
                            st.caption(f"⏱️ {event.get('timestamp', 'N/A')}")
                
                # 2. Tool Execution Section
                if tool_events:
                    with st.expander("🔧 Tool Execution", expanded=True):
                        for i, event in enumerate(tool_events, 1):
                            st.markdown(f"**Tool {i}: {event.get('tool_name', 'Unknown')}**")
                            
                            # Display metrics in columns
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Results", event.get("result_count", 0))
                            with col2:
                                st.metric("Time (s)", f"{event.get('execution_time', 0):.3f}")
                            with col3:
                                st.caption(f"⏱️ {event.get('timestamp', 'N/A')}")
                            
                            # Show query details
                            with st.expander(f"Query Details - Tool {i}"):
                                st.text("Query:")
                                st.code(event.get("query", "N/A"))
                                st.text("Cypher:")
                                st.code(event.get("cypher", "N/A"), language="cypher")
                                if "error" in event:
                                    st.error(f"Error: {event['error']}")
                            
                            st.divider()
                
                # 3. Reflection Section
                if reflection_events:
                    with st.expander("🤔 Reflection", expanded=True):
                        for event in reflection_events:
                            decision = event.get("decision", "UNKNOWN")
                            context_count = event.get("context_count", 0)
                            
                            if "YES" in decision:
                                st.success(f"✅ Sufficient information ({context_count} context items)")
                            else:
                                st.warning(f"⚠️ Need more information ({context_count} context items)")
                            
                            st.caption(f"⏱️ {event.get('timestamp', 'N/A')}")
                
                # 4. Answer Section
                if answer_events:
                    with st.expander("💡 Answer Generation", expanded=True):
                        for event in answer_events:
                            st.markdown(f"**Answer Length:** {event.get('answer_length', 0)} characters")
                            st.caption(f"⏱️ {event.get('timestamp', 'N/A')}")
                
                # Display execution summary
                if "execution_summary" in result:
                    st.divider()
                    st.subheader("📊 Execution Summary")
                    summary = result["execution_summary"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tools Called", summary.get("tools_called", 0))
                    with col2:
                        st.metric("Results Retrieved", summary.get("results_retrieved", 0))
                    with col3:
                        st.metric("Total Time (s)", summary.get("execution_time_seconds", 0))
                    
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

