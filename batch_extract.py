#!/usr/bin/env python3
"""
Batch extract responses from all scenario folders in a benchmarker results directory.

This script automatically finds all scenario subfolders and extracts responses from each.

Usage:
    python batch_extract.py <results_folder>

Example:
    python batch_extract.py data/results/extract_acta_files_8a204e2c/
"""

import json
import os
import sys
from pathlib import Path


def extract_responses_from_folder(folder_path, output_base_dir):
    """Extract responses from all JSON files in a single folder."""
    folder = Path(folder_path)
    
    # Create output directory
    output_dir = output_base_dir / folder.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all JSON files
    json_files = list(folder.glob("*.json"))
    
    if not json_files:
        return 0
    
    processed = 0
    
    for json_file in json_files:
        # Skip metadata files
        if json_file.name in ['scenario_summary.json', 'other_tasks.json']:
            continue
            
        try:
            # Load the JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract response
            if 'response' in data:
                response_data = data['response']
                
                # If response is a JSON string, parse it
                if isinstance(response_data, str):
                    try:
                        response_data = json.loads(response_data)
                    except json.JSONDecodeError:
                        pass  # Keep as string if not valid JSON
                
                # Save extracted response
                output_file = output_dir / f"response_{json_file.name}"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, indent=2, ensure_ascii=False)
                
                print(f"  ✓ {json_file.name} -> response_{json_file.name}")
                processed += 1
            else:
                print(f"  ✗ No 'response' field in: {json_file.name}")
                
        except Exception as e:
            print(f"  ✗ Error processing {json_file.name}: {e}")
    
    return processed


def batch_extract_responses(results_folder):
    """Extract responses from all scenario folders in a results directory."""
    results_path = Path(results_folder)
    
    if not results_path.exists():
        print(f"Error: Results folder '{results_folder}' does not exist")
        return
    
    if not results_path.is_dir():
        print(f"Error: '{results_folder}' is not a directory")
        return
    
    # Create main output directory
    output_base = results_path / "extracted_responses"
    output_base.mkdir(exist_ok=True)
    
    # Find all subdirectories that contain JSON files
    scenario_folders = []
    for item in results_path.iterdir():
        if item.is_dir() and item.name != "extracted_responses":
            json_files = list(item.glob("*.json"))
            # Filter out metadata files
            data_files = [f for f in json_files if f.name not in ['scenario_summary.json', 'other_tasks.json']]
            if data_files:
                scenario_folders.append(item)
    
    if not scenario_folders:
        print(f"No scenario folders with JSON data files found in '{results_folder}'")
        return
    
    total_processed = 0
    
    print(f"Found {len(scenario_folders)} scenario folders to process:")
    print()
    
    for folder in scenario_folders:
        print(f"Processing scenario: {folder.name}")
        processed = extract_responses_from_folder(folder, output_base)
        total_processed += processed
        print(f"  Processed {processed} files from {folder.name}")
        print()
    
    print(f"Summary:")
    print(f"  Total scenarios processed: {len(scenario_folders)}")
    print(f"  Total response files extracted: {total_processed}")
    print(f"  Results saved to: {output_base}")
    
    # Create index file
    index_file = output_base / "index.json"
    index_data = {
        "extraction_summary": {
            "source_folder": str(results_path),
            "total_scenarios": len(scenario_folders),
            "total_responses_extracted": total_processed,
            "scenarios": {}
        }
    }
    
    for folder in scenario_folders:
        response_files = list((output_base / folder.name).glob("response_*.json"))
        index_data["extraction_summary"]["scenarios"][folder.name] = {
            "response_count": len(response_files),
            "response_files": [f.name for f in response_files]
        }
    
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    print(f"  Index file created: {index_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python batch_extract.py <results_folder>")
        print("Example: python batch_extract.py data/results/extract_acta_files_8a204e2c/")
        sys.exit(1)
    
    batch_extract_responses(sys.argv[1])
