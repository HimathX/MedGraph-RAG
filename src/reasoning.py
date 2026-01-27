from typing import TypedDict, List, Annotated, Dict, Any
import operator
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from .llm import get_llm
from .retriever import HybridRetriever

class AgentState(TypedDict):
    """The detailed state of the reasoning agent."""
    query: str
    plan: List[str]
    current_step: int
    context: Annotated[List[dict], operator.add]
    answer: str
    reflection: str
    execution_events: Annotated[List[Dict[str, Any]], operator.add]
    execution_summary: Dict[str, Any]

class ReasoningAgent:
    def __init__(self):
        self.llm = get_llm()
        self.retriever = HybridRetriever()
        
        # Build Graph
        builder = StateGraph(AgentState)
        
        builder.add_node("plan", self.plan_node)
        builder.add_node("tool", self.tool_node)
        builder.add_node("reflect", self.reflect_node)
        builder.add_node("synthesis", self.synthesis_node)
        
        builder.set_entry_point("plan")
        
        builder.add_edge("plan", "tool")
        builder.add_edge("tool", "reflect")
        
        # Conditional edge: Reflect -> (Tool or Synthesis)
        builder.add_conditional_edges(
            "reflect",
            self.should_continue
        )
        builder.add_edge("synthesis", END)
        
        self.graph = builder.compile()

    async def plan_node(self, state: AgentState):
        """Gemini 3 decomposes the query."""
        print("--- PLAN ---")
        prompt = f"Break down this query into 2-3 step-by-step search tasks: {state['query']}"
        response = await self.llm.ainvoke(prompt)
        
        # Store the raw content for display
        raw_plan = response.content
        
        # Parse steps more intelligently - look for lines that start with "Step", "Task", or numbered patterns
        # For execution, we'll extract the main task titles
        lines = [s.strip() for s in raw_plan.split('\n') if s.strip()]
        steps = []
        for line in lines:
            # Extract lines that look like task headers (Step X:, Task X:, or start with numbers)
            if any(line.startswith(prefix) for prefix in ['Step ', 'Task ', '1.', '2.', '3.', '1)', '2)', '3)']):
                steps.append(line)
        
        # If we didn't find structured steps, fall back to the first 3 non-empty lines
        if not steps:
            steps = lines[:3]
        
        # Emit event with raw plan for display
        event = {
            "type": "plan_created",
            "timestamp": datetime.now().isoformat(),
            "plan": raw_plan  # Store as single string instead of list
        }
        
        return {"plan": steps, "current_step": 0, "execution_events": [event]}

    async def tool_node(self, state: AgentState):
        """Executes the current step using Hybrid Retriever."""
        print("--- TOOL ---")
        step_idx = state.get("current_step", 0)
        if step_idx >= len(state["plan"]):
            return {"context": [], "execution_events": []} # Should probably go to synthesis

        current_task = state["plan"][step_idx]
        print(f"Executing: {current_task}")
        
        # Decide Local vs Global (simplified: just do Hybrid)
        # Result is now a List[Dict] containing structured docs with metadata
        results, metadata_list = await self.retriever.retrieve(current_task)
        
        # Emit tool events for each retrieval method
        events = []
        for metadata in metadata_list:
            event = {
                "type": "tool_call",
                "tool_name": "vector_search" if "vector" in str(metadata.get("cypher", "")) else "community_search",
                "query": metadata.get("query", ""),
                "cypher": metadata.get("cypher", ""),
                "result_count": metadata.get("result_count", 0),
                "execution_time": metadata.get("execution_time", 0),
                "timestamp": metadata.get("timestamp", datetime.now().isoformat())
            }
            if "error" in metadata:
                event["error"] = metadata["error"]
            events.append(event)
        
        return {"context": results, "execution_events": events}

    async def reflect_node(self, state: AgentState):
        """Reflects on whether we have enough info."""
        print("--- REFLECT ---")
        context_str = "\n".join([f"[{c.get('source', 'UNKNOWN').upper()}] {c.get('content', '')}" for c in state["context"]])
        prompt = f"""
        Query: {state['query']}
        Current Context: {context_str}
        
        Do we have enough information to answer the query? 
        Reply YES or NO.
        """
        response = await self.llm.ainvoke(prompt)
        reflection = response.content.strip().upper()
        
        # Emit event
        event = {
            "type": "reflection",
            "timestamp": datetime.now().isoformat(),
            "decision": reflection,
            "context_count": len(state["context"])
        }
        
        return {"reflection": reflection, "current_step": state["current_step"] + 1, "execution_events": [event]}

    def should_continue(self, state: AgentState):
        """Decides direction based on reflection."""
        if "YES" in state["reflection"] or state["current_step"] >= len(state["plan"]):
            return "synthesis"
        return "tool"

    async def synthesis_node(self, state: AgentState):
        """Generates the final answer."""
        print("--- SYNTHESIS ---")
        context_str = "\n".join([f"[{c.get('source', 'UNKNOWN').upper()}] {c.get('content', '')}" for c in state["context"]])
        prompt = f"""
        Answer the query based strictly on the context.
        Query: {state['query']}
        Context: {context_str}
        """
        response = await self.llm.ainvoke(prompt)
        
        # Emit event
        event = {
            "type": "final_answer",
            "timestamp": datetime.now().isoformat(),
            "answer_length": len(response.content)
        }
        
        return {"answer": response.content, "execution_events": [event]}

    async def run(self, query: str):
        inputs = {
            "query": query, 
            "plan": [], 
            "current_step": 0, 
            "context": [], 
            "answer": "", 
            "reflection": "",
            "execution_events": [],
            "execution_summary": {}
        }
        result = await self.graph.ainvoke(inputs)
        
        # Compute execution summary
        tool_events = [e for e in result.get("execution_events", []) if e.get("type") == "tool_call"]
        total_results = sum(e.get("result_count", 0) for e in tool_events)
        total_time = sum(e.get("execution_time", 0) for e in tool_events)
        
        result["execution_summary"] = {
            "tools_called": len(tool_events),
            "results_retrieved": total_results,
            "execution_time_seconds": round(total_time, 2)
        }
        
        return result
