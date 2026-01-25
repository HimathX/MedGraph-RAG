from typing import TypedDict, List, Annotated
import operator
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
        # Simple parsing - assuming LLM gives newline separated steps
        steps = [s.strip() for s in response.content.split('\n') if s.strip()]
        return {"plan": steps, "current_step": 0}

    async def tool_node(self, state: AgentState):
        """Executes the current step using Hybrid Retriever."""
        print("--- TOOL ---")
        step_idx = state.get("current_step", 0)
        if step_idx >= len(state["plan"]):
            return {"context": []} # Should probably go to synthesis

        current_task = state["plan"][step_idx]
        print(f"Executing: {current_task}")
        
        # Decide Local vs Global (simplified: just do Hybrid)
        # Result is now a List[Dict] containing structured docs with metadata
        results = await self.retriever.retrieve(current_task)
        
        return {"context": results}

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
        return {"reflection": reflection, "current_step": state["current_step"] + 1}

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
        return {"answer": response.content}

    async def run(self, query: str):
        inputs = {"query": query, "plan": [], "current_step": 0, "context": [], "answer": "", "reflection": ""}
        return await self.graph.ainvoke(inputs)
