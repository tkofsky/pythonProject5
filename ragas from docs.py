import pandas as pd
from ragas import evaluate

# Sample Testing Data
test_set = [
    {"question": "What is the capital of France?", "generated_answer": "Paris", "expected_answer": "Paris"},
    {"question": "Who wrote '1984'?", "generated_answer": "George Orwell", "expected_answer": "George Orwell"}

]

# Convert the test set into a DataFrame
df = pd.DataFrame(test_set)

print(df.columns)  # This should output: Index(['question', 'generated_answer', 'expected_answer'], dtype='object')


# Display the DataFrame to confirm correct structure
print(df)

# Evaluate the results using RAGAS
results = evaluate(df)

# Display the results
print(results)
