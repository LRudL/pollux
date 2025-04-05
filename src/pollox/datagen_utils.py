import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def get_docs_from_source(source: str, prefix: str="") -> str:
    """Assume source is a path; load all .txt files starting with prefix.
    Return a list of strings, each representing the contents of a .txt file.
    """
    return [
        open(os.path.join(source, f), encoding="utf-8").read()
        for f in os.listdir(source)
        if f.startswith(prefix) and f.endswith(".txt")
    ]

@dataclass
class FTDoc:
    content: str

@dataclass
class FTQA:
    question: str
    answer: str

@dataclass
class DPOPair:
    question: str
    answer: str

