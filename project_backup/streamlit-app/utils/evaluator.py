"""
Small evaluation set using SQuAD and CoQA style Q&A pairs.
The evaluation context is embedded and used for retrieval.
"""
from typing import List, Dict, Tuple
import re
import string

# ---- Sample evaluation data ----

EVAL_CONTEXT = """
Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering 
research on radioactivity. She was the first woman to win a Nobel Prize, the first person to win 
the Nobel Prize twice, and the only person to win the Nobel Prize in two different sciences 
(Physics in 1903 and Chemistry in 1911). She was born in Warsaw, Poland, on November 7, 1867. 
She moved to Paris in 1891 to study at the University of Paris, where she later became the 
first woman professor. Together with her husband Pierre Curie, she discovered the elements 
polonium and radium. She died on July 4, 1934, of aplastic anemia, believed to be caused by 
her long-term exposure to radiation.

The Python programming language was created by Guido van Rossum and first released in 1991. 
Python emphasizes code readability and simplicity, using significant whitespace. It supports 
multiple programming paradigms including procedural, object-oriented, and functional programming. 
Python is widely used in data science, machine learning, web development, and automation. 
The language's design philosophy is captured in "The Zen of Python" document which emphasizes 
beautiful, explicit, simple, and readable code. Python 3 was released in 2008 and is not fully 
backward-compatible with Python 2.

The Great Wall of China is a series of fortifications built across the historical northern borders 
of ancient Chinese states and Imperial China as protection against raids from nomadic groups. 
The wall stretches for thousands of miles, though the precise length varies by definition and 
measurement method. The most well-preserved and visited sections were built during the Ming Dynasty 
(1368-1644). The Great Wall is listed as a UNESCO World Heritage Site and is one of the most 
recognized symbols of China worldwide. Construction began as early as the 7th century BC.
"""

EVAL_QA = [
    # SQuAD-style
    {
        "source": "SQuAD",
        "question": "Who was Marie Curie?",
        "expected": "a Polish and naturalized-French physicist and chemist",
    },
    {
        "source": "SQuAD",
        "question": "How many Nobel Prizes did Marie Curie win?",
        "expected": "two",
    },
    {
        "source": "SQuAD",
        "question": "What elements did Marie and Pierre Curie discover?",
        "expected": "polonium and radium",
    },
    # CoQA-style (conversational)
    {
        "source": "CoQA",
        "question": "Who created Python?",
        "expected": "Guido van Rossum",
    },
    {
        "source": "CoQA",
        "question": "When was Python first released?",
        "expected": "1991",
    },
    {
        "source": "CoQA",
        "question": "What dynasty built the most well-preserved sections of the Great Wall?",
        "expected": "Ming Dynasty",
    },
    {
        "source": "SQuAD",
        "question": "What caused Marie Curie's death?",
        "expected": "aplastic anemia",
    },
    {
        "source": "CoQA",
        "question": "What is The Zen of Python about?",
        "expected": "beautiful, explicit, simple, and readable code",
    },
]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_text(prediction) == normalize_text(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(truth_tokens) if truth_tokens else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def partial_match(prediction: str, ground_truth: str) -> bool:
    """True if expected keywords appear in the prediction."""
    key_words = normalize_text(ground_truth).split()
    pred_norm = normalize_text(prediction)
    return all(w in pred_norm for w in key_words)
