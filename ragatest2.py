from deepeval import evaluate
from deepeval.metrics.ragas import RagasMetric
from deepeval.test_case import LLMTestCase
import asyncio

# Replace this with the actual output from your LLM application
actual_output = "We offer a 30-day full refund at no extra cost."

# Replace this with the expected output from your RAG generator
expected_output = "You are eligible for a 30 day full refund at no extra cost."

# Replace this with the actual retrieved context from your RAG pipeline
retrieval_context = ["All customers are eligible for a 30 day full refund at no extra cost."]

metric = RagasMetric(threshold=0.5, model="gpt-3.5-turbo")
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=retrieval_context
)

metric.measure(test_case)
print(metric.score)

# or evaluate test cases in bulk
evaluate([test_case], [metric])