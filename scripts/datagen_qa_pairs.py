import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer
from pollox.datagen_utils import get_docs_from_source, client, inspect_dataset

def generate_questions(doc: str, n: Optional[int] = None) -> List[str]:
    """Generate questions from a document using Claude."""
    prompt = f"""Given the following document, generate a list of questions that this document has answers to.

More about the type of question:
- not questions about specific facts, but questions about takes, judgements, or analysis
- imagine the types of questions you might want to ask the expert who wrote this doc (before you read the doc and found the expert's take in it)
- imagine that the author is an expert, and wrote this doc in response to being asked questions; what might these questions have been?

Document:
{doc}

More about format:
The questions should be in a machine-parsable format as a JSON array of strings.

Please generate questions in this exact format:
["Question 1?", "Question 2?", ...]
"""

    response = client.chat.completions.create(
        model="anthropic/claude-3.5-sonnet",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    
    try:
        # Find the first '[' in the response
        content = response.choices[0].message.content
        start_idx = content.find('[')
        if start_idx == -1:
            print("Warning: No JSON array found in Claude's response")
            return []
            
        # Find the matching closing bracket
        bracket_count = 1
        end_idx = start_idx + 1
        while bracket_count > 0 and end_idx < len(content):
            if content[end_idx] == '[':
                bracket_count += 1
            elif content[end_idx] == ']':
                bracket_count -= 1
            end_idx += 1
            
        if bracket_count > 0:
            print("Warning: Could not find matching closing bracket in Claude's response")
            return []
            
        # Extract and parse the JSON array
        json_str = content[start_idx:end_idx]
        questions = json.loads(json_str)
        
        if n is not None:
            questions = questions[:n]
        return questions
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Warning: Failed to parse Claude's response: {e}")
        return []

def generate_qa_pairs(doc: str, questions: List[str]) -> List[Dict[str, str]]:
    """Generate QA pairs for a document and its questions using Claude."""
    qa_pairs = []
    
    for question in tqdm(questions, desc="Generating answers"):
        prompt = f"""Given the following document and question, generate a concise answer based on the document's content, paying particular attention to the types of takes, analyses, and judgements that the expert author of the document makes.

Document:
{doc}

Question: {question}

Please provide your answer below, as if you were the expert author of the document responding to the above question. Do NOT say "based on the document"; answer in the voice of the document's author directly.
"""

        response = client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
        )
        
        answer = response.choices[0].message.content.strip()
        qa_pairs.append({
            "question": question,
            "answer": answer
        })
    
    return qa_pairs

def create_chat_format(qa_pair: Dict[str, str]) -> List[Dict[str, str]]:
    """Convert a QA pair into the Gemma chat format."""
    return [
        {"role": "user", "content": qa_pair["question"]},
        {"role": "model", "content": qa_pair["answer"]}
    ]

def process_documents(
    doc_source: str = "duncan-docs",
    prefix: str = "Blog-Post-",
    n: Optional[int] = None,
    phases: List[str] = ["questions", "qa", "hf"]
) -> None:
    """Process documents through the QA generation pipeline."""
    # Create necessary directories
    os.makedirs("datasets/intermediate/questions", exist_ok=True)
    os.makedirs("datasets/intermediate/qa", exist_ok=True)
    os.makedirs("datasets/qa", exist_ok=True)
    
    # Get documents
    docs = get_docs_from_source(doc_source, prefix=prefix)
    if n is not None:
        docs = docs[:n]
    
    # Phase 1: Question Generation
    if "questions" in phases:
        print("Phase 1: Generating questions...")
        for i, doc in enumerate(tqdm(docs)):
            questions = generate_questions(doc)
            output_file = f"datasets/intermediate/questions/doc_{i}.json"
            with open(output_file, "w") as f:
                json.dump(questions, f, indent=2)
    
    # Phase 2: QA Pair Generation
    if "qa" in phases:
        print("Phase 2: Generating QA pairs...")
        for i, doc in enumerate(tqdm(docs)):
            # Load questions
            with open(f"datasets/intermediate/questions/doc_{i}.json", "r") as f:
                questions = json.load(f)
            
            qa_pairs = generate_qa_pairs(doc, questions)
            output_file = f"datasets/intermediate/qa/doc_{i}.json"
            with open(output_file, "w") as f:
                json.dump(qa_pairs, f, indent=2)
    
    # Phase 3: HF Dataset Creation
    if "hf" in phases:
        print("Phase 3: Creating HuggingFace dataset...")
        # Check if intermediate files exist
        if not os.path.exists("datasets/intermediate/qa"):
            print("Error: QA pairs not found. Please run the 'qa' phase first to generate QA pairs.")
            return
            
        # Load all QA pairs
        all_qa_pairs = []
        for i in range(len(docs)):
            qa_file = f"datasets/intermediate/qa/doc_{i}.json"
            if not os.path.exists(qa_file):
                print(f"Error: Missing QA pairs file {qa_file}. Please run the 'qa' phase first.")
                return
            with open(qa_file, "r") as f:
                all_qa_pairs.extend(json.load(f))
        
        # Initialize tokenizer and set up chat template
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
            "{% elif message['role'] == 'model' %}"
            "<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n"
            "{% endif %}"
            "{% endfor %}"
        )
        
        # Create chat format and tokenize
        dataset_dict = {
            "chat": [create_chat_format(qa) for qa in all_qa_pairs]
        }
        dataset = Dataset.from_dict(dataset_dict)
        
        # Tokenize the dataset
        def tokenize_chat(example):
            chat = example["chat"]
            # Add BOS token at the start
            prompt = "<bos>" + tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            
            # Create labels with -100 for user messages
            labels = tokenized["input_ids"][0].clone()
            user_tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": chat[0]["content"]}],
                tokenize=False,
                add_generation_prompt=True
            )
            user_tokenized = tokenizer(user_tokens, return_tensors="pt", add_special_tokens=False)
            labels[:len(user_tokenized["input_ids"][0])] = -100
            
            return {
                "input_ids": tokenized["input_ids"][0].tolist(),
                "labels": labels.tolist(),
                "chat": chat  # Keep the original chat format
            }
        
        tokenized_dataset = dataset.map(
            tokenize_chat,
            desc="Tokenizing dataset"  # Removed remove_columns parameter to keep chat column
        )
        
        # Save dataset
        tokenized_dataset.save_to_disk("datasets/qa")
        print(f"Saved tokenized dataset with {len(tokenized_dataset)} examples")

def main():
    parser = argparse.ArgumentParser(description='Generate QA pairs from documents')
    parser.add_argument('-n', type=int, help='Number of documents to process')
    parser.add_argument('-questions', action='store_true', help='Run question generation phase')
    parser.add_argument('-qa', action='store_true', help='Run QA pair generation phase')
    parser.add_argument('-hf', action='store_true', help='Run HuggingFace dataset creation phase')
    parser.add_argument('-inspect', action='store_true', help='Inspect the existing dataset instead of generating')
    args = parser.parse_args()
    
    if args.inspect:
        inspect_dataset("datasets/qa")
        return
    
    # Determine which phases to run
    phases = []
    if args.questions or args.qa or args.hf:
        if args.questions:
            phases.append("questions")
        if args.qa:
            phases.append("qa")
        if args.hf:
            phases.append("hf")
    else:
        # If no phases specified, run all
        phases = ["questions", "qa", "hf"]
    
    process_documents(n=args.n, phases=phases)

if __name__ == "__main__":
    main()

