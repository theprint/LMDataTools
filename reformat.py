# reformat.py
import json
import os
import sys
from tqdm import tqdm
from datacore.io.json_ops import DatasetFormatter

def main():
    """Main function to run the reformatting process."""
    try:
        # Load config from the job directory
        with open("config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)

        target_format = config.get("target_format", "alpaca")
        dataset_name = config.get("dataset_name", "reformatted-dataset")
        import_path = config.get("import_path", "import")

        # Find the uploaded file
        input_file = None
        if os.path.isdir(import_path):
            for file in os.listdir(import_path):
                if file.endswith(('.json', '.jsonl')):
                    input_file = os.path.join(import_path, file)
                    break
        
        if not input_file:
            print("ERROR: No .json or .jsonl file found in the import directory.", file=sys.stderr)
            sys.exit(1)

        # Load data (handles both json and jsonl)
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            try:
                # Try loading as a single JSON object first
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("JSON is not a list of objects.")
            except (json.JSONDecodeError, ValueError):
                # If that fails, treat it as JSONL
                f.seek(0)
                data = [json.loads(line) for line in f if line.strip()]

        if not data:
            print("ERROR: Input file is empty or not in a valid JSON/JSONL format.", file=sys.stderr)
            sys.exit(1)

        # --- Helper to detect format ---
        def detect_format(entry):
            if "conversations" in entry and isinstance(entry["conversations"], list): return "sharegpt"
            if "instruction" in entry and "output" in entry: return "alpaca"
            if "question" in entry and "answer" in entry: return "qa"
            return "unknown"

        # --- Validation Step 1: Detect source format ---
        source_format = detect_format(data[0])
        if source_format == "unknown" or not source_format:
            print(f"ERROR: The source file does not appear to be in a supported format (Alpaca, ShareGPT, or Q&A).", file=sys.stderr)
            sys.exit(1)

        print(f"Detected source format: {source_format}")

        # --- Validation Step 2: Check if source and target formats are the same ---
        if source_format == target_format:
            print(f"ERROR: Source format ({source_format}) is the same as the target format ({target_format}). No conversion needed.", file=sys.stderr)
            sys.exit(1)
        
        # --- Initialize Formatter from datacore ---
        formatter = DatasetFormatter(from_format=source_format, to_format=target_format)

        # --- Conversion Process ---
        output_data = []
        total_entries = len(data)
        for i, entry in enumerate(tqdm(data, desc="Reformatting entries")):
            converted = formatter.reformat_entry(entry)
            if converted:
                output_data.append(converted)
            # The progress pattern in webapp.py expects this format
            print(f"Reformatted entry {i + 1} of {total_entries}")
            sys.stdout.flush()

        # Remove the unwanted '_reformatted' flag from each entry
        for entry in output_data:
            entry.pop('_reformatted', None)

        # Save the converted data
        output_filename = f"{dataset_name}.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        print(f"Successfully reformatted {len(output_data)} entries.")
        print(f"Output saved to {output_filename}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
