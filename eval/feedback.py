import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file to instantiate the client
load_dotenv()

client = OpenAI()


def groundedness_score(source: str, statement: str) -> float:
    """
    Computes groundedness of a statement given a source, scaled to 0-1.
    """
    prompt = f"""You are a fact-checking expert. Given the SOURCE text, determine if the STATEMENT is entirely supported by the information in the SOURCE.
Respond with a single number from 0 to 10, where 0 means 'not grounded' and 10 means 'perfectly grounded'.

SOURCE:
{source}

STATEMENT:
{statement}

SCORE (0-10):
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5,
    )

    try:
        score_str = response.choices[0].message.content.strip()
        # Find the first number in the response string
        match = re.search(r"\d+\.?\d*", score_str)
        if match:
            score = float(match.group())
            return min(max(score / 10.0, 0.0), 1.0)  # Scale to 0-1
    except (ValueError, IndexError):
        return 0.0
    return 0.0


def relevance_score(question: str, text: str) -> float:
    """
    Computes relevance of a text to a question, scaled to 0-1.
    Used for both context relevance and answer relevance.
    """
    prompt = f"""You are a relevance-scoring expert. Given the QUESTION, determine how relevant the TEXT is to answering the question.
Respond with a single number from 0 to 10, where 0 means 'not relevant' and 10 means 'perfectly relevant'.

QUESTION:
{question}

TEXT:
{text}

SCORE (0-10):
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5,
    )

    try:
        score_str = response.choices[0].message.content.strip()
        match = re.search(r"\d+\.?\d*", score_str)
        if match:
            score = float(match.group())
            return min(max(score / 10.0, 0.0), 1.0)  # Scale to 0-1
    except (ValueError, IndexError):
        return 0.0
    return 0.0
