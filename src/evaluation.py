"""
RAGAS Evaluation Module for MedGraph-RAG

This module provides evaluation capabilities using RAGAS metrics to measure:
- Faithfulness: Ensures answers are grounded in the retrieved context
- Answer Relevance: Measures how relevant the answer is to the question
- Context Precision: Evaluates the quality of retrieved context
- Context Recall: Measures if all relevant information was retrieved
"""

from typing import List, Dict, Any
import pandas as pd
from datetime import datetime


class RAGASEvaluator:
    """Evaluator for RAG system using RAGAS metrics"""
    
    def __init__(self):
        """Initialize the RAGAS evaluator with default metrics"""
        # Don't import at init time to avoid circular dependencies
        self._ragas_imported = False
        self._evaluate = None
        self._faithfulness = None
        self._answer_relevancy = None
        self._context_precision = None
        self._context_recall = None
        self._Dataset = None
    
    def _ensure_ragas_imported(self):
        """Lazy import of RAGAS to avoid circular dependency deadlock"""
        if not self._ragas_imported:
            try:
                from ragas import evaluate
                from ragas.metrics import (
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                )
                from datasets import Dataset
                from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
                import os
                
                # Configure RAGAS to use Google Gemini instead of OpenAI
                # Get the Google API key from environment
                google_api_key = os.getenv("GOOGLE_API_KEY")
                if not google_api_key:
                    raise ValueError("GOOGLE_API_KEY environment variable not set")
                
                # Initialize Gemini LLM and embeddings for RAGAS
                self._llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    google_api_key=google_api_key,
                    temperature=0
                )
                
                self._embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=google_api_key
                )
                
                self._evaluate = evaluate
                self._faithfulness = faithfulness
                self._answer_relevancy = answer_relevancy
                self._context_precision = context_precision
                self._context_recall = context_recall
                self._Dataset = Dataset
                self._ragas_imported = True
            except Exception as e:
                raise ImportError(f"Failed to import RAGAS: {e}. Please ensure RAGAS is properly installed.")
    
    def prepare_evaluation_data(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ):
        """
        Prepare data in the format required by RAGAS
        
        Args:
            questions: List of user queries
            answers: List of generated answers
            contexts: List of retrieved context chunks (each is a list of strings)
            ground_truths: Optional list of reference answers for context recall
            
        Returns:
            Dataset object ready for RAGAS evaluation
        """
        self._ensure_ragas_imported()
        
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        if ground_truths:
            data["ground_truth"] = ground_truths
        
        return self._Dataset.from_dict(data)
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None
    ) -> Dict[str, float]:
        """
        Evaluate a single question-answer pair
        
        Args:
            question: User query
            answer: Generated answer
            contexts: Retrieved context chunks
            ground_truth: Optional reference answer
            
        Returns:
            Dictionary of metric scores
        """
        self._ensure_ragas_imported()
        
        # Prepare data
        questions = [question]
        answers = [answer]
        contexts_list = [contexts]
        
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        }
        
        if ground_truth:
            dataset_dict["ground_truth"] = [ground_truth]
        
        dataset = self._Dataset.from_dict(dataset_dict)
        
        # Select metrics based on available data
        metrics_to_use = [self._faithfulness, self._answer_relevancy]
        if ground_truth:
            metrics_to_use.extend([self._context_precision, self._context_recall])
        
        # Run evaluation with Gemini LLM and embeddings
        result = self._evaluate(
            dataset, 
            metrics=metrics_to_use,
            llm=self._llm,
            embeddings=self._embeddings
        )
        
        # Convert result to dictionary format for easier access
        result_dict = {}
        if hasattr(result, 'to_pandas'):
            # Convert to pandas and extract first row as dict
            df = result.to_pandas()
            if not df.empty:
                result_dict = df.iloc[0].to_dict()
        elif hasattr(result, '__dict__'):
            # Try to extract attributes
            result_dict = {k: v for k, v in result.__dict__.items() if not k.startswith('_')}
        
        return result_dict
    
    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> pd.DataFrame:
        """
        Evaluate multiple question-answer pairs
        
        Args:
            questions: List of user queries
            answers: List of generated answers
            contexts: List of retrieved context chunks
            ground_truths: Optional list of reference answers
            
        Returns:
            DataFrame with evaluation results
        """
        self._ensure_ragas_imported()
        
        dataset = self.prepare_evaluation_data(
            questions, answers, contexts, ground_truths
        )
        
        # Select metrics based on available data
        metrics_to_use = [self._faithfulness, self._answer_relevancy]
        if ground_truths:
            metrics_to_use.extend([self._context_precision, self._context_recall])
        
        # Run evaluation with Gemini LLM and embeddings
        result = self._evaluate(
            dataset, 
            metrics=metrics_to_use,
            llm=self._llm,
            embeddings=self._embeddings
        )
        
        return result.to_pandas()
    
    def get_hallucination_score(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Get a hallucination score (inverse of faithfulness)
        
        Args:
            question: User query
            answer: Generated answer
            contexts: Retrieved context chunks
            
        Returns:
            Hallucination score (0-1, lower is better)
        """
        result = self.evaluate_single(question, answer, contexts)
        faithfulness_score = result.get('faithfulness', 0.0)
        
        # Hallucination score is inverse of faithfulness
        return 1.0 - faithfulness_score
    
    def is_answer_faithful(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        threshold: float = 0.7
    ) -> bool:
        """
        Check if an answer is faithful to the context
        
        Args:
            question: User query
            answer: Generated answer
            contexts: Retrieved context chunks
            threshold: Minimum faithfulness score (default: 0.7)
            
        Returns:
            True if answer is faithful, False otherwise
        """
        result = self.evaluate_single(question, answer, contexts)
        faithfulness_score = result.get('faithfulness', 0.0)
        
        return faithfulness_score >= threshold


class EvaluationLogger:
    """Logger for storing evaluation results"""
    
    def __init__(self, log_file: str = "data/evaluation_logs.csv"):
        """Initialize the logger with a file path"""
        self.log_file = log_file
    
    def log_evaluation(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        metrics: Dict[str, float],
        metadata: Dict[str, Any] = None
    ):
        """
        Log an evaluation result
        
        Args:
            question: User query
            answer: Generated answer
            contexts: Retrieved context chunks
            metrics: Evaluation metrics
            metadata: Optional additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "num_contexts": len(contexts),
            **metrics,
        }
        
        if metadata:
            log_entry.update(metadata)
        
        # Append to CSV
        df = pd.DataFrame([log_entry])
        try:
            existing_df = pd.read_csv(self.log_file)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass
        
        df.to_csv(self.log_file, index=False)
    
    def get_evaluation_history(self) -> pd.DataFrame:
        """Get all evaluation history"""
        try:
            return pd.read_csv(self.log_file)
        except FileNotFoundError:
            return pd.DataFrame()
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics across all evaluations"""
        df = self.get_evaluation_history()
        if df.empty:
            return {}
        
        metric_columns = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
        available_metrics = [col for col in metric_columns if col in df.columns]
        
        return df[available_metrics].mean().to_dict()
