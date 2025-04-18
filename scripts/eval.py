import os
import json
import random
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict
from tqdm import tqdm
import string

# Load environment variables
load_dotenv()

def create_win_rate_graph(comparison_file):
    """Create a bar chart of model win rates from comparison.json"""
    with open(comparison_file, "r") as f:
        comparison = json.load(f)
    
    # Extract model names and win rates
    stats = comparison["statistics"]
    models = []
    win_rates = []
    
    for model, data in stats.items():
        # Clean up model name by removing prefixes/suffixes
        clean_name = model.replace("results_", "").replace(".json", "")
        models.append(clean_name)
        win_rates.append(data["percentage"])
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, win_rates)
    
    # Customize the chart
    plt.title("Model Win Rates (%)", pad=20)
    plt.xlabel("Model")
    plt.ylabel("Win Rate (%)")
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    output_dir = comparison_file.parent
    plt.savefig(output_dir / "win_rates.png", format='png', bbox_inches='tight')
    plt.close()
    
    print(f"Wrote graph to {output_dir / 'win_rates.png'}")

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def load_questions():
    """Load questions and expert answers from qs.json"""
    with open("datasets/duncanbench/qs.json", "r") as f:
        questions = json.load(f)
    return {q["id"]: {"question": q["content"], "expert": q["expert"]} for q in questions}

def load_results(results_file):
    """Load results from a results_*.json file"""
    with open(results_file, "r") as f:
        data = json.load(f)
    return {i+1: result["answer"] for i, result in enumerate(data["data"])}

def extract_letter(response):
    """Extract the first valid letter from Claude's response"""
    # Clean up the response
    response = response.strip()
    
    # Try to find the first letter in the response
    for char in response:
        if char in string.ascii_uppercase:
            return char
    return None

def get_claude_judgement(question, answers):
    """Get Claude's judgement on which answer is best"""
    # Get list of sources and their answers
    sources = list(answers.keys())
    answer_texts = [answers[source] for source in sources]
    
    # Create a random mapping from letters to indices
    num_answers = len(sources)
    letters = list(string.ascii_uppercase[:num_answers])
    random.shuffle(letters)
    
    # Create prompt with blinded answers
    prompt = f"""Question: {question}

Answers:
"""
    for letter, answer in zip(letters, answer_texts):
        prompt += f"\n{letter}. {answer}\n"
    
    prompt += "\nWhich answer is best? Respond with ONLY the letter of the best answer and nothing else."
    
    # Get Claude's response
    completion = client.chat.completions.create(
        model="anthropic/claude-3.5-sonnet",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Extract letter from response
    response = completion.choices[0].message.content.strip()
    letter = extract_letter(response)
    
    if not letter or letter not in letters:
        return None, prompt, response
    
    # Map letter back to source
    letter_to_idx = {letter: idx for idx, letter in enumerate(letters)}
    return sources[letter_to_idx[letter]], prompt, response

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run model evaluation or generate graphs')
    parser.add_argument('-graph', action='store_true', help='Only generate graphs from existing comparison.json')
    args = parser.parse_args()
    
    results_dir = Path("datasets/duncanbench")
    comparison_file = results_dir / "comparison.json"
    
    # If -graph flag is set, only generate graphs
    if args.graph:
        if not comparison_file.exists():
            print("Error: comparison.json not found. Run evaluation first.")
            return
        create_win_rate_graph(comparison_file)
        return
    
    # Load questions and expert answers
    questions = load_questions()
    
    # Get all results files
    results_files = list(results_dir.glob("results_*.json"))
    
    # Initialize statistics with all possible sources
    stats = defaultdict(lambda: {"total": 0})
    stats["expert"] = {"total": 0}  # Initialize expert
    for results_file in results_files:
        stats[f"results_{results_file.stem.split('_', 1)[1]}"] = {"total": 0}
    
    # Store all judgements
    judgements = []
    
    # Process each question with progress bar
    for q_id, q_data in tqdm(questions.items(), desc="Processing questions"):
        question = q_data["question"]
        
        # Collect all answers including expert
        answers = {"expert": q_data["expert"]}
        
        # Add answers from each results file with progress bar
        for results_file in tqdm(results_files, desc=f"Loading results for question {q_id}", leave=False):
            results = load_results(results_file)
            if q_id in results:
                answers[f"results_{results_file.stem.split('_', 1)[1]}"] = results[q_id]
        
        # Get Claude's judgement
        best_source, prompt, claude_response = get_claude_judgement(question, answers)
        
        # Store the judgement details regardless of whether it was valid
        judgements.append({
            "question_id": q_id,
            "question": question,
            "answers": answers,
            "prompt": prompt,
            "claude_response": claude_response,
            "best_source": best_source
        })
        
        # Only count valid judgements in statistics
        if best_source:
            stats[best_source]["total"] += 1
    
    # Write comparison statistics
    comparison = {
        "statistics": {
            source: {
                "wins": stats[source]["total"],
                "percentage": round(stats[source]["total"] / len(questions) * 100, 2)
            }
            for source in stats
        },
        "judgements": judgements
    }
    
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Wrote out to {comparison_file}")
    
    # Generate graphs
    create_win_rate_graph(comparison_file)

if __name__ == "__main__":
    main()
