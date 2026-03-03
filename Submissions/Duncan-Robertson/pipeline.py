# Example structure:
import sys
from pathlib import Path
# Add project root to path so we can import src
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.Scoring import score
from data_loader import load_papers

def extract_sdrf(paper_text):
    """Your extraction logic here"""
    pass

def has_test_labels():
    return Path('test_solution.csv').exists()

def main():
    # Load data
    papers = load_papers()

    # Extract metadata
    predictions = [extract_sdrf(p) for p in papers]

    # Generate submission
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv('submission.csv', index=False)

    # (Optional) Score locally if you have test labels
    if has_test_labels():
        solution_df = pd.read_csv('test_solution.csv')
        eval_df, score = score(solution_df, submission_df, 'ID')
        print(f"Local F1 Score: {score:.6f}")

if __name__ == "__main__":
    main()