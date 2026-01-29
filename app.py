import streamlit as st
import asyncio
import os
from src.reasoning import ReasoningAgent
from streamlit_agraph import agraph, Node, Edge, Config
from src.graph import Neo4jManager

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

# Evidence Cloud: Render interactive graph
def render_evidence_graph(context):
    """
    Creates an interactive graph visualization of entities and relationships
    discovered during retrieval.
    """
    neo4j = Neo4jManager()
    
    # Combine all retrieved content (lowercase for matching)
    all_content = " ".join([doc.get("content", "") for doc in context]).lower()
    
    if not all_content.strip():
        st.info("No content to analyze for entities.")
        neo4j.close()
        return
    
    # Query all entities from the graph
    entity_query = """
    MATCH (e:Entity)
    RETURN e.name as entity_name
    LIMIT 200
    """
    
    matched_entities = []
    entity_canonical = {}  # Map lowercase -> original case
    
    try:
        with neo4j.driver.session() as session:
            result = session.run(entity_query)
            all_entities = [r["entity_name"] for r in result]
            
            # Filter to entities that appear in the content
            # Fix case sensitivity: use lowercase for matching, keep first occurrence
            for entity in all_entities:
                if entity and len(entity) > 2:
                    entity_lower = entity.lower()
                    if entity_lower in all_content and entity_lower not in entity_canonical:
                        entity_canonical[entity_lower] = entity
                        matched_entities.append(entity)
            
            matched_entities = matched_entities[:30]
            
    except Exception as e:
        st.warning(f"Could not fetch entities: {e}")
        neo4j.close()
        return
    
    if not matched_entities:
        st.info("No graph entities found in the retrieved content.")
        neo4j.close()
        return
    
    # Add legend
    st.markdown("""
    <div style="display: flex; gap: 20px; margin-bottom: 10px;">
        <span style="display: flex; align-items: center; gap: 5px;">
            <span style="width: 12px; height: 12px; background: #FF6B6B; border: 2px solid #000; display: inline-block;"></span>
            <span style="font-size: 12px;">Entities from your sources</span>
        </span>
        <span style="display: flex; align-items: center; gap: 5px;">
            <span style="width: 12px; height: 12px; background: #4ECDC4; border: 2px solid #000; display: inline-block;"></span>
            <span style="font-size: 12px;">Connected entities (discovered)</span>
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption(f"Found {len(matched_entities)} entities: {', '.join(matched_entities[:5])}{'...' if len(matched_entities) > 5 else ''}")
    
    # Query 2: Find relationships with ACTUAL relationship types
    relationship_query = """
    MATCH (a:Entity)-[r]->(b:Entity)
    WHERE (a.name IN $entities OR b.name IN $entities)
    AND type(r) <> 'RELATED_TO'
    RETURN a.name as source, b.name as target, type(r) as rel_type
    LIMIT 60
    UNION ALL
    MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
    WHERE (a.name IN $entities OR b.name IN $entities)
    RETURN a.name as source, b.name as target, COALESCE(r.type, 'RELATED') as rel_type
    LIMIT 40
    """
    
    nodes = []
    edges = []
    node_set = set()  # Use lowercase for deduplication
    node_display = {}  # lowercase -> display name
    matched_set = set([e.lower() for e in matched_entities])
    
    try:
        with neo4j.driver.session() as session:
            result = session.run(relationship_query, entities=matched_entities)
            
            for record in result:
                source = record["source"]
                target = record["target"]
                rel_type = record.get("rel_type", "RELATED") or "RELATED"
                
                source_lower = source.lower()
                target_lower = target.lower()
                
                # Deduplicate by lowercase
                if source_lower not in node_set:
                    color = "#FF6B6B" if source_lower in matched_set else "#4ECDC4"
                    nodes.append(Node(
                        id=source_lower, 
                        label=source, 
                        size=30,
                        color=color,
                        font={"color": "#FFFFFF", "size": 14, "strokeWidth": 2, "strokeColor": "#000000"}
                    ))
                    node_set.add(source_lower)
                    node_display[source_lower] = source
                    
                if target_lower not in node_set:
                    color = "#FF6B6B" if target_lower in matched_set else "#4ECDC4"
                    nodes.append(Node(
                        id=target_lower, 
                        label=target, 
                        size=30,
                        color=color,
                        font={"color": "#FFFFFF", "size": 14, "strokeWidth": 2, "strokeColor": "#000000"}
                    ))
                    node_set.add(target_lower)
                    node_display[target_lower] = target
                
                # Add edge with actual relationship type
                edges.append(Edge(
                    source=source_lower, 
                    target=target_lower, 
                    label=rel_type,
                    color="#FFD93D",
                    font={"color": "#FFD93D", "size": 12, "strokeWidth": 1, "strokeColor": "#000"}
                ))
    
    except Exception as e:
        st.warning(f"Could not fetch relationships: {e}")
        neo4j.close()
        return
    finally:
        neo4j.close()
    
    if not nodes:
        st.info(f"No relationships found for the {len(matched_entities)} entities.")
        return
    
    # Configure graph with Neo-Brutalist styling
    config = Config(
        width=750,
        height=550,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#FFD93D",
        collapsible=False,
    )
    
    st.caption(f"Showing {len(nodes)} entities and {len(edges)} relationships")
    agraph(nodes=nodes, edges=edges, config=config)


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
                            # Display the plan as-is (it's now a single string)
                            plan_content = event.get("plan", "")
                            if isinstance(plan_content, list):
                                # Fallback for old format
                                plan_content = "\n".join(plan_content)
                            st.markdown(plan_content)
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
                
                # Source Inspector: Side-by-Side Layout
                st.divider()
                
                # Initialize selected citation in session state
                if "selected_citation" not in st.session_state:
                    st.session_state.selected_citation = None
                
                # Create two columns: 70% for answer, 30% for source inspector
                col_answer, col_inspector = st.columns([7, 3])
                
                with col_answer:
                    st.subheader("💡 Answer")
                    answer = result.get("answer", "No answer generated.")
                    
                    # Parse and make citations clickable
                    import re
                    citation_pattern = r'\[(\d+)\]'
                    
                    # Find all citations
                    citations = re.findall(citation_pattern, answer)
                    
                    # Display answer with clickable citations
                    if citations:
                        # Split by citations and rebuild with buttons
                        parts = re.split(citation_pattern, answer)
                        
                        # Display the answer with inline citation buttons
                        st.markdown(answer)
                        
                        # Display citation buttons below
                        st.caption("Click a citation number to view source:")
                        citation_cols = st.columns(min(len(set(citations)), 10))
                        for idx, cite_num in enumerate(sorted(set(citations), key=int)):
                            with citation_cols[idx % 10]:
                                if st.button(f"[{cite_num}]", key=f"cite_{cite_num}"):
                                    st.session_state.selected_citation = int(cite_num) - 1
                                    st.rerun()
                    else:
                        st.markdown(answer)
                
                with col_inspector:
                    st.subheader("📄 Source Inspector")
                    
                    if st.session_state.selected_citation is not None:
                        cite_idx = st.session_state.selected_citation
                        context = result.get("context", [])
                        
                        if 0 <= cite_idx < len(context):
                            doc = context[cite_idx]
                            
                            st.markdown(f"**Source [{cite_idx + 1}]**")
                            st.caption(f"📚 {doc.get('metadata', {}).get('doc_title', 'Unknown')}")
                            st.caption(f"📑 {doc.get('metadata', {}).get('section_title', 'Unknown')}")
                            
                            st.divider()
                            
                            # Display content in a scrollable container
                            st.markdown("**Content:**")
                            st.text_area(
                                "Source Content",
                                doc.get('content', 'No content available'),
                                height=300,
                                label_visibility="collapsed"
                            )
                            
                            if st.button("Clear", key="clear_citation"):
                                st.session_state.selected_citation = None
                                st.rerun()
                        else:
                            st.info(f"Citation [{cite_idx + 1}] not found in sources.")
                    else:
                        st.info("👈 Click a citation number in the answer to view its source here.")
                
                # Evidence Cloud (Interactive Graph)
                if "context" in result and result["context"]:
                    st.divider()
                    st.subheader("🕸️ Evidence Cloud")
                    st.caption("Interactive visualization of the knowledge graph supporting this answer")
                    render_evidence_graph(result["context"])
                
                # Discovery Suggestions (Follow-up Queries)
                if "followup_queries" in result and result["followup_queries"]:
                    st.divider()
                    st.subheader("🔍 Explore Further")
                    st.caption("Suggested follow-up questions to explore the knowledge graph:")
                    
                    # Display as clickable buttons
                    for idx, query in enumerate(result["followup_queries"]):
                        if st.button(f"💡 {query}", key=f"followup_{idx}"):
                            # Store the selected query in session state
                            st.session_state.suggested_query = query
                            st.rerun()
                
                # Display Citations (Full List)
                if "context" in result and result["context"]:
                    st.divider()
                    st.subheader("📚 All Sources")
                    for idx, doc in enumerate(result["context"]):
                        with st.expander(f"Source {idx+1}: {doc.get('metadata', {}).get('doc_title', 'Unknown')}"):
                            st.caption(f"Section: {doc.get('metadata', {}).get('section_title', 'Unknown')}")
                            st.text(doc.get('content'))
                
                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                status.update(label="❌ Failed", state="error")

# Handle suggested query clicks
if "suggested_query" in st.session_state and st.session_state.suggested_query:
    # Auto-submit the suggested query
    suggested = st.session_state.suggested_query
    st.session_state.suggested_query = None  # Clear it
    
    # Add to messages
    st.session_state.messages.append({"role": "user", "content": suggested})
    st.rerun()


