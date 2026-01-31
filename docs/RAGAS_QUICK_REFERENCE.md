# RAGAS Integration Quick Reference

## 🚀 Quick Start

### 1. Enable RAGAS in Streamlit UI
```
1. Run: streamlit run app.py
2. Open sidebar
3. Check "Enable Real-time Evaluation"
4. Adjust faithfulness threshold (default: 0.7)
5. Ask a question and see evaluation below the answer
```

### 2. View Evaluation Dashboard
```
1. Navigate to "evaluation_dashboard" page in Streamlit
2. Or run: streamlit run pages/evaluation_dashboard.py
3. View metrics, trends, and historical data
```

### 3. Programmatic Usage
```python
from src.evaluation import RAGASEvaluator

evaluator = RAGASEvaluator()
result = evaluator.evaluate_single(
    question="Your question",
    answer="Generated answer",
    contexts=["Context 1", "Context 2"]
)

print(f"Faithfulness: {result['faithfulness']:.3f}")
print(f"Relevancy: {result['answer_relevancy']:.3f}")
```

## 📊 Metrics Explained

| Metric | Range | What it Measures | Good Score |
|--------|-------|------------------|------------|
| **Faithfulness** | 0-1 | Answer consistency with context (anti-hallucination) | ≥ 0.7 |
| **Answer Relevancy** | 0-1 | How well answer addresses the question | ≥ 0.7 |
| **Context Precision** | 0-1 | Quality of retrieved context | ≥ 0.7 |
| **Context Recall** | 0-1 | Completeness of retrieved context | ≥ 0.7 |

## 🎯 Score Interpretation

### Faithfulness
- **0.9 - 1.0**: Excellent - Fully grounded in sources ✅
- **0.7 - 0.9**: Good - Mostly grounded, minor issues ✅
- **0.5 - 0.7**: Moderate - Some unsupported claims ⚠️
- **< 0.5**: Poor - Significant hallucination ❌

### Answer Relevancy
- **0.9 - 1.0**: Excellent - Directly answers the question ✅
- **0.7 - 0.9**: Good - Relevant with minor tangents ✅
- **0.5 - 0.7**: Moderate - Partially relevant ⚠️
- **< 0.5**: Poor - Off-topic or irrelevant ❌

## 🔧 Common Tasks

### Check for Hallucination
```python
from src.evaluation import RAGASEvaluator

evaluator = RAGASEvaluator()
is_faithful = evaluator.is_answer_faithful(
    question="...",
    answer="...",
    contexts=["..."],
    threshold=0.7
)

if not is_faithful:
    print("⚠️ Hallucination detected!")
```

### Batch Evaluate
```python
evaluator = RAGASEvaluator()
results_df = evaluator.evaluate_batch(
    questions=["Q1", "Q2", "Q3"],
    answers=["A1", "A2", "A3"],
    contexts=[["C1"], ["C2"], ["C3"]]
)
print(results_df)
```

### Log Evaluation
```python
from src.evaluation import EvaluationLogger

logger = EvaluationLogger()
logger.log_evaluation(
    question="...",
    answer="...",
    contexts=["..."],
    metrics={"faithfulness": 0.85, "answer_relevancy": 0.92},
    metadata={"model": "gemini-2.0-flash"}
)
```

### Get Average Metrics
```python
logger = EvaluationLogger()
avg_metrics = logger.get_average_metrics()
print(f"Avg Faithfulness: {avg_metrics['faithfulness']:.3f}")
```

## 🎨 UI Components

### Sidebar Controls
- **Enable Real-time Evaluation**: Toggle RAGAS on/off
- **Faithfulness Threshold**: Set minimum acceptable score
- **Show Detailed Metrics**: Display full breakdown
- **Log Evaluations**: Save to CSV

### Evaluation Display
- Color-coded scores (green/yellow/red)
- Hallucination warnings
- Detailed metrics in expander
- Interpretation guide

### Dashboard Tabs
1. **Overview**: Summary metrics and timeline
2. **Metrics Analysis**: Distributions and correlations
3. **Query Explorer**: Browse individual evaluations
4. **Raw Data**: Export and analyze

## ⚙️ Configuration

### Set Faithfulness Threshold
```python
# In UI: Use slider in sidebar
# Programmatically:
is_faithful = evaluator.is_answer_faithful(
    question="...",
    answer="...",
    contexts=["..."],
    threshold=0.8  # Stricter threshold
)
```

### Custom Metrics
```python
# Edit src/evaluation.py
from ragas.metrics import context_relevancy

class RAGASEvaluator:
    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_relevancy,  # Add custom metric
        ]
```

## 📁 File Structure

```
MedGraph-RAG/
├── src/
│   └── evaluation.py          # Core evaluation module
├── pages/
│   └── evaluation_dashboard.py # Dashboard page
├── examples/
│   └── ragas_evaluation_example.py # Usage examples
├── docs/
│   └── RAGAS_INTEGRATION_GUIDE.md  # Full guide
└── data/
    └── evaluation_logs.csv    # Logged evaluations
```

## 🐛 Troubleshooting

### Issue: Slow evaluation
**Solution**: Disable real-time evaluation, use batch mode offline

### Issue: Low faithfulness scores
**Causes**: Poor retrieval, irrelevant context, LLM hallucination
**Solutions**: Review retrieval quality, improve graph traversal

### Issue: Evaluation errors
**Check**: API keys, context format, non-empty contexts

## 📚 Resources

- Full Guide: `docs/RAGAS_INTEGRATION_GUIDE.md`
- Examples: `examples/ragas_evaluation_example.py`
- RAGAS Docs: https://docs.ragas.io/
- RAGAS Paper: https://arxiv.org/abs/2309.15217

## 💡 Best Practices

1. **Set appropriate thresholds** based on use case:
   - Critical medical: 0.8+
   - General use: 0.7 (recommended)
   - Exploratory: 0.5-0.6

2. **Enable logging** for production queries

3. **Review dashboard regularly** to track trends

4. **Use batch evaluation** for regression testing

5. **Monitor hallucination rate** (% with faithfulness < 0.7)

## 🎓 Example Workflow

```python
# 1. Run agent
from src.reasoning import ReasoningAgent
agent = ReasoningAgent()
result = await agent.run("Your question")

# 2. Extract data
question = "Your question"
answer = result["answer"]
contexts = [doc["content"] for doc in result["context"]]

# 3. Evaluate
from src.evaluation import RAGASEvaluator
evaluator = RAGASEvaluator()
eval_result = evaluator.evaluate_single(question, answer, contexts)

# 4. Check quality
if eval_result['faithfulness'] < 0.7:
    print("⚠️ Warning: Low faithfulness!")
    
# 5. Log result
from src.evaluation import EvaluationLogger
logger = EvaluationLogger()
logger.log_evaluation(question, answer, contexts, eval_result)

# 6. View in dashboard
# Navigate to evaluation_dashboard page
```

## 🔑 Key Takeaways

✅ RAGAS prevents hallucination by measuring faithfulness
✅ Real-time evaluation in Streamlit UI
✅ Historical tracking via dashboard
✅ Programmatic API for custom workflows
✅ Logging for analysis and monitoring
✅ Threshold-based quality gates

---

**Need help?** See `docs/RAGAS_INTEGRATION_GUIDE.md` for detailed documentation.
