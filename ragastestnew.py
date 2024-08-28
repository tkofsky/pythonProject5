from datasets import Dataset
import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness

data_samples = {
    'question': [
        'When was the first super bowl?',
        'Who won the most super bowls?'
    ],
    'answer': [
        'The first superbowl was held on Jan 15, 1967',
        'The most super bowls have been won by The New England wqwqwqwq'
    ],
    'contexts': [
        [
            'The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'
        ],
        [
            'The Green Bay Packers...Green Bay, Wisconsin.',
            'The Packers compete...Football Conference'
        ]
    ],
    'ground_truth': [
        'The first superbowl was held on January 15, 1969',
        'The New England Patriots have won the Super Bowl a record five times'
    ]
}

dataset = Dataset.from_dict(data_samples)

score = evaluate(dataset, metrics=[faithfulness, answer_correctness])
df = score.to_pandas()
df.to_csv('score.csv', index=False)