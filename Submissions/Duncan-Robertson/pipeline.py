import argparse
import json
import os
import re
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PROMPT_PATH = SCRIPT_DIR / "data" / "BaselinePrompt.txt"
OUTPUT_DIR = SCRIPT_DIR / "data" / "Gemini-Extract"

# Load environment variables
load_dotenv(SCRIPT_DIR / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def extract_sdrf(pub_json_path: Path):
    """Extract metadata using Gemini 2.5 Flash."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

    # 1. Parse PXD ID from filename (e.g., PXD004010_PubText.json -> PXD004010)
    filename = pub_json_path.name
    pxd_match = re.search(r'(PXD\d+)', filename)
    if not pxd_match:
        print(f"Could not find PXD number in {filename}")
        return
    pxd_id = pxd_match.group(1)

    # 2. Read the prompt
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Could not find prompt file: {PROMPT_PATH}")
        
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    # 3. Read the publication JSON
    with open(pub_json_path, 'r', encoding='utf-8') as f:
        pub_data = json.load(f)

    # Format manuscript text and raw files to match the prompt's expected input
    manuscript_parts = []
    for section in ["TITLE", "ABSTRACT", "METHODS"]:
        if section in pub_data and pub_data[section]:
            manuscript_parts.append(f"--- {section} ---\n{pub_data[section]}")
    
    manuscript_text = "\n\n".join(manuscript_parts)
    raw_files = pub_data.get("Raw Data Files", [])
    
    user_content = f"MANUSCRIPT_TEXT:\n{manuscript_text}\n\nRAW_FILES:\n{json.dumps(raw_files, indent=2)}"

    # 4. Call Gemini API
    print(f"Processing {pxd_id} with gemini-2.5-flash...")
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction=system_prompt
    )
    
    try:
        # Force strict JSON output using generation_config
        response = model.generate_content(
            user_content,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0  # Low temperature for strict factual extraction
            )
        )
        output_json = response.text
    except Exception as e:
        print(f"Error calling Gemini API for {pxd_id}: {e}")
        return

    # Validate JSON output
    try:
        parsed_json = json.loads(output_json)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON output for {pxd_id}: {e}")
        print(f"Raw output was:\n{output_json}")
        return

    # Inject PXD ID into each raw file's metadata
    enriched_json = {}
    for raw_file, metadata in parsed_json.items():
        if isinstance(metadata, dict):
            new_metadata = {"PXD": [pxd_id]}
            new_metadata.update(metadata)
            enriched_json[raw_file] = new_metadata
        else:
            enriched_json[raw_file] = metadata
    parsed_json = enriched_json

    # 5. Save the output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{pxd_id}_metadata.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_json, f, indent=2)
        
    print(f"Saved extraction to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Extract SDRF metadata from a PXD publication JSON.")
    parser.add_argument(
        "pxd_id", 
        nargs="?", 
        default="PXD000070", 
        help="The PXD ID to process (e.g., PXD004010). Default is PXD000070."
    )
    args = parser.parse_args()

    pxd_id = args.pxd_id
    
    # Check both Training and Test directories
    train_file = REPO_ROOT / "data" / "TrainingPubText" / f"{pxd_id}_PubText.json"
    test_file = REPO_ROOT / "data" / "TestPubText" / f"{pxd_id}_PubText.json"
    
    if train_file.exists():
        extract_sdrf(train_file)
    elif test_file.exists():
        extract_sdrf(test_file)
    else:
        print(f"Could not find a PubText JSON file for {pxd_id} in either TrainingPubText or TestPubText directories.")

if __name__ == "__main__":
    main()
