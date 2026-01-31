# RAGAS Integration Guide for MedGraph-RAG

## Overview

This guide explains how to use RAGAS (Retrieval Augmented Generation Assessment) evaluation in your MedGraph-RAG Streamlit application. RAGAS helps ensure your system doesn't hallucinate outside the provided 15 PDFs by measuring faithfulness and answer relevance.

## What is RAGAS?

RAGAS is an evaluation framework specifically designed for RAG systems. It provides several key metrics:

- **Faithfulness**: Measures whether the answer is factually consistent with the retrieved context (prevents hallucination)
- **Answer Relevancy**: Measures how well the answer addresses the user's question
- **Context Precision**: Evaluates the quality of retrieved context
- **Context Recall**: Measures if all relevant information was retrieved

## Installation

RAGAS is already included in your `pyproject.toml`:

```toml
dependencies = [
    "ragas>=0.2.0",
    # ... other dependencies
]
```

Make sure to install/sync dependencies:

```bash
uv sync
```

## Architecture

### 1. Evaluation Module (`src/evaluation.py`)

The core evaluation module provides:

- **`RAGASEvaluator`**: Main class for running evaluations
  - `evaluate_single()`: Evaluate a single Q&A pair
  - `evaluate_batch()`: Evaluate multiple Q&A pairs
  - `get_hallucination_score()`: Get inverse of faithfulness
  - `is_answer_faithful()`: Boolean check for faithfulness

- **`EvaluationLogger`**: Logging and persistence
  - `log_evaluation()`: Save evaluation results to CSV
  - `get_evaluation_history()`: Retrieve past evaluations
  - `get_average_metrics()`: Calculate average metrics

### 2. Main App Integration (`app.py`)

RAGAS evaluation is integrated into the main Streamlit app with:

#### Sidebar Controls
- **Enable Real-time Evaluation**: Toggle RAGAS on/off
- **Faithfulness Threshold**: Set minimum acceptable faithfulness (default: 0.7)
- **Show Detailed Metrics**: Display full metric breakdown
- **Log Evaluations**: Save results to CSV for analysis

#### Evaluation Display
After each answer is generated, if RAGAS is enabled:
1. Extracts context from retrieved documents
2. Runs evaluation using RAGAS metrics
3. Displays color-coded scores:
   - ✅ Green: Score ≥ 0.7 (Good)
   - ⚠️ Yellow: Score 0.5-0.7 (Moderate)
   - ❌ Red: Score < 0.5 (Poor)
4. Shows hallucination warnings if faithfulness is low
5. Optionally logs results to `data/evaluation_logs.csv`

### 3. Evaluation Dashboard (`pages/evaluation_dashboard.py`)

A dedicated dashboard for analyzing evaluation metrics over time:

#### Features
- **Overview Tab**: Summary metrics and timeline visualization
- **Metrics Analysis Tab**: Distribution plots and correlation analysis
- **Query Explorer Tab**: Browse and filter individual evaluations
- **Raw Data Tab**: View and export raw evaluation data

## Usage Guide

### Basic Usage

1. **Start the main app**:
   ```bash
   streamlit run app.py
   ```

2. **Enable RAGAS evaluation**:
   - Open the sidebar
   - Check "Enable Real-time Evaluation"
   - Adjust the faithfulness threshold if needed (default: 0.7)
   - Enable "Show Detailed Metrics" for full breakdown
   - Enable "Log Evaluations" to save results

3. **Ask a question**:
   - Type your medical question in the chat input
   - Wait for the answer to be generated
   - Review the RAGAS evaluation section below the answer

4. **Interpret the results**:
   - **High Faithfulness (≥0.7)**: Answer is well-grounded in sources ✅
   - **Low Faithfulness (<0.7)**: Potential hallucination detected ⚠️
   - **High Relevancy (≥0.7)**: Answer addresses the question well ✅
   - **Low Relevancy (<0.5)**: Answer may be off-topic ❌

### Viewing Historical Metrics

1. **Access the dashboard**:
   - Navigate to the evaluation dashboard page in Streamlit
   - Or run: `streamlit run pages/evaluation_dashboard.py`

2. **Filter data**:
   - Use the sidebar to filter by date range
   - Set minimum thresholds for metrics
   - View filtered results in real-time

3. **Analyze trends**:
   - **Overview**: See metrics over time
   - **Metrics Analysis**: View distributions and correlations
   - **Query Explorer**: Examine individual queries
   - **Raw Data**: Export data for external analysis

### Programmatic Usage

You can also use the evaluation module programmatically:

```python
from src.evaluation import RAGASEvaluator

# Initialize evaluator
evaluator = RAGASEvaluator()

# Evaluate a single answer
result = evaluator.evaluate_single(
    question="What are the side effects of Drug X?",
    answer="Drug X may cause headaches and nausea.",
    contexts=[
        "Drug X clinical trials showed headaches in 10% of patients.",
        "Common side effects include nausea and dizziness."
    ]
)

print(f"Faithfulness: {result['faithfulness']:.3f}")
print(f"Answer Relevancy: {result['answer_relevancy']:.3f}")

# Check if answer is faithful
is_faithful = evaluator.is_answer_faithful(
    question="What are the side effects of Drug X?",
    answer="Drug X may cause headaches and nausea.",
    contexts=[...],
    threshold=0.7
)

if not is_faithful:
    print("Warning: Potential hallucination detected!")
```

### Batch Evaluation

For evaluating multiple Q&A pairs:

```python
from src.evaluation import RAGASEvaluator

evaluator = RAGASEvaluator()

questions = [
    "What is the mechanism of action?",
    "What are the contraindications?"
]

answers = [
    "The drug works by inhibiting...",
    "Contraindications include..."
]

contexts = [
    ["Context for question 1..."],
    ["Context for question 2..."]
]

# Evaluate batch
results_df = evaluator.evaluate_batch(questions, answers, contexts)
print(results_df)
```

## Metrics Interpretation

### Faithfulness (0-1)
- **What it measures**: Whether the answer is factually consistent with the context
- **How it works**: Uses LLM to extract claims from the answer and verify them against the context
- **Threshold**: ≥0.7 is considered good
- **Use case**: Detecting hallucinations

### Answer Relevancy (0-1)
- **What it measures**: How well the answer addresses the question
- **How it works**: Uses LLM to generate questions from the answer and compares to original
- **Threshold**: ≥0.7 is considered good
- **Use case**: Ensuring answers are on-topic

### Context Precision (0-1)
- **What it measures**: Quality of retrieved context
- **How it works**: Checks if relevant context appears early in the list
- **Requires**: Ground truth answers
- **Use case**: Optimizing retrieval

### Context Recall (0-1)
- **What it measures**: Whether all relevant information was retrieved
- **How it works**: Checks if ground truth can be attributed to context
- **Requires**: Ground truth answers
- **Use case**: Ensuring comprehensive retrieval

## Best Practices

### 1. Setting Thresholds
- **Conservative (0.8+)**: For critical medical applications
- **Balanced (0.7)**: For general use (recommended)
- **Lenient (0.5-0.6)**: For exploratory queries

### 2. Handling Low Scores
- **Low Faithfulness**: 
  - Review the retrieved context
  - Check if relevant documents are in the knowledge base
  - Consider improving entity extraction or graph traversal
  
- **Low Relevancy**:
  - Review the question understanding
  - Check if the retrieval is finding relevant context
  - Consider improving the query planning

### 3. Logging and Monitoring
- Enable logging for all production queries
- Regularly review the evaluation dashboard
- Track metrics over time to identify degradation
- Use batch evaluation for regression testing

### 4. Performance Considerations
- RAGAS evaluation adds 2-5 seconds per query
- Disable for real-time demos if latency is critical
- Use batch evaluation for offline analysis
- Consider caching evaluation results

## Troubleshooting

### Issue: Evaluation is slow
**Solution**: RAGAS uses LLM calls for evaluation. Consider:
- Using a faster model for evaluation
- Disabling real-time evaluation and using batch mode
- Reducing the number of contexts evaluated

### Issue: Low faithfulness scores across the board
**Possible causes**:
- Retrieved context doesn't support the answers
- Graph traversal is finding irrelevant entities
- LLM is adding information not in the context

**Solutions**:
- Review retrieval quality
- Adjust hybrid search weights
- Improve entity extraction and normalization

### Issue: Evaluation fails with errors
**Common causes**:
- Missing API keys for LLM
- Invalid context format
- Empty contexts

**Solutions**:
- Check environment variables
- Ensure contexts are non-empty strings
- Review error messages in the UI

## Advanced Usage

### Custom Metrics

You can add custom metrics by modifying `src/evaluation.py`:

```python
from ragas.metrics import context_relevancy

class RAGASEvaluator:
    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_relevancy,  # Add custom metric
        ]
```

### Integration with CI/CD

Create a test suite for regression testing:

```python
# tests/test_evaluation.py
import pytest
from src.evaluation import RAGASEvaluator

def test_answer_faithfulness():
    evaluator = RAGASEvaluator()
    
    result = evaluator.evaluate_single(
        question="Test question",
        answer="Test answer",
        contexts=["Test context"]
    )
    
    assert result['faithfulness'] >= 0.7, "Faithfulness below threshold"
```

### Export for Analysis

Export evaluation logs for external analysis:

```python
from src.evaluation import EvaluationLogger
import pandas as pd

logger = EvaluationLogger()
df = logger.get_evaluation_history()

# Export to various formats
df.to_csv('evaluations.csv')
df.to_json('evaluations.json')
df.to_excel('evaluations.xlsx')

# Analyze in pandas
print(df.describe())
print(df.groupby('model')['faithfulness'].mean())
```

## References

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- [RAGAS Paper](https://arxiv.org/abs/2309.15217)

## Summary

RAGAS integration in MedGraph-RAG provides:
- ✅ Real-time hallucination detection
- ✅ Answer quality assessment
- ✅ Historical metrics tracking
- ✅ Visual analytics dashboard
- ✅ Programmatic evaluation API

This ensures your system stays grounded in the 15 PDFs and doesn't generate unsupported claims.
