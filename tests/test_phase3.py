import asyncio
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph import Neo4jManager
from src.llm import get_embeddings, get_llm
from src.reasoning import ReasoningAgent

async def run_test():
    print(">>> Setting up Test Environment...")
    neo4j = Neo4jManager()
    embeddings = get_embeddings()
    llm = get_llm()
    
    # 1. Setup Vector Index
    print(">>> Creating Vector Index...")
    neo4j.create_vector_index(dimension=768) # Assuming 768 for text-embedding-005
    
    # 2. Seed Dummy Data (Chunk with Embedding)
    print(">>> Seeding Dummy Data...")
    dummy_text = "Hypertension is a chronic medical condition formed by high blood pressure in the arteries."
    dummy_embedding = embeddings.embed_query(dummy_text)
    
    seed_query = """
    MERGE (c:Chunk {id: 'dummy_1'})
    SET c.content = $text,
        c.embedding = $embedding,
        c.source = 'test_doc'
    """
    with neo4j.driver.session() as session:
        session.run(seed_query, text=dummy_text, embedding=dummy_embedding)
        
    # 3. Run Reasoner
    query = "What is Hypertension?"
    print(f">>> Running Reasoning Agent for query: '{query}'...")
    agent = ReasoningAgent()
    result = await agent.run(query)
    
    final_answer = result["answer"]
    context = result["context"]
    print(f"\n>>> Final Answer: {final_answer}\n")
    
    # 4. AI Judge
    print(">>> Running AI Judge Verification...")
    judge_prompt = f"""
    You are an impartial AI Judge.
    Query: {query}
    Retrieved Context: {context}
    Generated Answer: {final_answer}
    
    Task:
    1. Verify if the Answer is supported by the Context.
    2. Check for hallucinations.
    
    Output JSON: {{ "supported": bool, "reason": str }}
    """
    judge_response = await llm.ainvoke(judge_prompt)
    print(f"Judge Verdict: {judge_response.content}")
    
    # 5. Cost Monitoring (Simulated)
    # in real usage, we'd inspect response.usage_metadata
    print("\n>>> Cost Monitoring:")
    print("Estimated Input Tokens: ~500")
    print("Estimated Output Tokens: ~100")
    print("Estimated Cost: $0.0001 (Gemini 3 Pro Pricing)")

    neo4j.close()

if __name__ == "__main__":
    asyncio.run(run_test())
