"""
Example script demonstrating RAGAS evaluation usage

This script shows how to:
1. Evaluate a single Q&A pair
2. Batch evaluate multiple pairs
3. Check for hallucinations
4. Log evaluation results
"""

import asyncio
from src.evaluation import RAGASEvaluator, EvaluationLogger
from src.reasoning import ReasoningAgent


async def example_single_evaluation():
    """Example: Evaluate a single question-answer pair"""
    print("=" * 60)
    print("Example 1: Single Evaluation")
    print("=" * 60)
    
    evaluator = RAGASEvaluator()
    
    question = "What are the therapeutic effects of tau-targeting PROTACs?"
    answer = """Tau-targeting PROTACs have shown promising therapeutic effects in preclinical studies. 
    They effectively reduce both total tau and phosphorylated tau (p-tau) levels in cellular models. 
    For example, PROTAC C8 decreased tau levels in HEK293-hTau cells and improved cognitive function 
    in mouse models, as demonstrated in novel object recognition and Morris water maze tests."""
    
    contexts = [
        """C8 effectively decreased total tau and ptau levels in HEK293 cells stably expressing 
        WT full-length human tau (referred to as HEK293-hTau). In this mouse model, C8 treatment 
        (10 mg/kg, i.p., 3 times/week for 1 month) ameliorated cognitive dysfunction in novel 
        object recognition and Morris water maze and reduced total tau and p-tau levels in the hippocampus.""",
        
        """The PROTAC C8 was designed using thalidomide as an E3 ligase binder, a long-chain alkyl 
        group was used as a linker, and quinoxaline attached a thiophene ring as tau binder. 
        This quinoxaline derivative tau binder has high affinity and selectivity for tau aggregates."""
    ]
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer}")
    print(f"\nNumber of contexts: {len(contexts)}")
    
    print("\nEvaluating...")
    result = evaluator.evaluate_single(question, answer, contexts)
    
    print("\n--- Evaluation Results ---")
    print(f"Faithfulness: {result.get('faithfulness', 0):.3f}")
    print(f"Answer Relevancy: {result.get('answer_relevancy', 0):.3f}")
    
    # Check for hallucination
    is_faithful = evaluator.is_answer_faithful(question, answer, contexts, threshold=0.7)
    if is_faithful:
        print("\n✅ Answer is faithful to the context (no hallucination detected)")
    else:
        print("\n⚠️ Warning: Potential hallucination detected!")
    
    print()


async def example_batch_evaluation():
    """Example: Batch evaluate multiple Q&A pairs"""
    print("=" * 60)
    print("Example 2: Batch Evaluation")
    print("=" * 60)
    
    evaluator = RAGASEvaluator()
    
    questions = [
        "What is the DC50 of PROTAC C8?",
        "Which E3 ligase does PROTAC C8 target?",
        "Does PROTAC C8 cross the blood-brain barrier?"
    ]
    
    answers = [
        "The DC50 of PROTAC C8 is 0.05 µM in cellular models.",
        "PROTAC C8 targets the CRBN E3 ligase.",
        "Yes, PROTAC C8 showed good blood-brain barrier permeability in BALB/C nude mice."
    ]
    
    contexts = [
        ["The DC50 of PROTAC C8 is 0.05µM in HEK293-hTau cell models."],
        ["PROTAC C8 uses CRBN as the target E3 ligase for tau degradation."],
        ["C8 showed good blood-brain barrier (BBB) permeability in BALB/C nude mice by in vivo imaging after intravenous injection."]
    ]
    
    print(f"\nEvaluating {len(questions)} Q&A pairs...")
    results_df = evaluator.evaluate_batch(questions, answers, contexts)
    
    print("\n--- Batch Evaluation Results ---")
    print(results_df[['faithfulness', 'answer_relevancy']])
    
    print("\n--- Summary Statistics ---")
    print(f"Average Faithfulness: {results_df['faithfulness'].mean():.3f}")
    print(f"Average Relevancy: {results_df['answer_relevancy'].mean():.3f}")
    print(f"Min Faithfulness: {results_df['faithfulness'].min():.3f}")
    print(f"Max Faithfulness: {results_df['faithfulness'].max():.3f}")
    
    print()


async def example_with_logging():
    """Example: Evaluate and log results"""
    print("=" * 60)
    print("Example 3: Evaluation with Logging")
    print("=" * 60)
    
    evaluator = RAGASEvaluator()
    logger = EvaluationLogger()
    
    question = "What animal models were used to test PROTAC C8?"
    answer = "PROTAC C8 was tested in C57BL/6 mice overexpressing human tau (hTau)."
    contexts = [
        "Purified pAAV-hSyn-hTau-mCHERRY-3×FLAG virus was injected bilaterally into the hippocampus CA1 region of male C57BL/6 mice (8 weeks old) to induce hTau-overexpressed mouse model."
    ]
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer}")
    
    print("\nEvaluating...")
    result = evaluator.evaluate_single(question, answer, contexts)
    
    print("\n--- Evaluation Results ---")
    print(f"Faithfulness: {result.get('faithfulness', 0):.3f}")
    print(f"Answer Relevancy: {result.get('answer_relevancy', 0):.3f}")
    
    # Log the evaluation
    print("\nLogging evaluation...")
    logger.log_evaluation(
        question=question,
        answer=answer,
        contexts=contexts,
        metrics=result,
        metadata={
            "model": "gemini-2.0-flash",
            "num_sources": len(contexts)
        }
    )
    
    print("✅ Evaluation logged to data/evaluation_logs.csv")
    
    # Show evaluation history
    history = logger.get_evaluation_history()
    if not history.empty:
        print(f"\nTotal evaluations in history: {len(history)}")
        avg_metrics = logger.get_average_metrics()
        print("\nAverage metrics across all evaluations:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.3f}")
    
    print()


async def example_with_real_agent():
    """Example: Evaluate answers from the actual reasoning agent"""
    print("=" * 60)
    print("Example 4: Evaluation with Real Agent")
    print("=" * 60)
    
    agent = ReasoningAgent()
    evaluator = RAGASEvaluator()
    
    question = "What are the main challenges in developing tau-targeting PROTACs?"
    
    print(f"\nQuestion: {question}")
    print("\nRunning agent...")
    
    result = await agent.run(question)
    
    answer = result.get("answer", "")
    context_docs = result.get("context", [])
    
    print(f"\nAnswer: {answer[:200]}...")
    print(f"\nRetrieved {len(context_docs)} context documents")
    
    # Extract context texts
    context_texts = [doc.get("content", "") for doc in context_docs]
    
    print("\nEvaluating agent's answer...")
    eval_result = evaluator.evaluate_single(question, answer, context_texts)
    
    print("\n--- Evaluation Results ---")
    print(f"Faithfulness: {eval_result.get('faithfulness', 0):.3f}")
    print(f"Answer Relevancy: {eval_result.get('answer_relevancy', 0):.3f}")
    
    # Hallucination check
    hallucination_score = evaluator.get_hallucination_score(question, answer, context_texts)
    print(f"Hallucination Score: {hallucination_score:.3f} (lower is better)")
    
    if hallucination_score > 0.3:
        print("\n⚠️ Warning: High hallucination risk!")
    else:
        print("\n✅ Low hallucination risk")
    
    print()


async def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("RAGAS Evaluation Examples for MedGraph-RAG")
    print("=" * 60 + "\n")
    
    # Run examples
    await example_single_evaluation()
    await example_batch_evaluation()
    await example_with_logging()
    
    # Uncomment to test with real agent (requires Neo4j connection)
    # await example_with_real_agent()
    
    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
