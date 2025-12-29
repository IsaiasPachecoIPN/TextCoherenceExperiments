import pandas as pd
import sys
import os
import argparse

# Add the utils directory to path. 
# Adjust "module_dir" if your folder structure is different.
module_dir = "../utils"
sys.path.append(os.path.abspath(module_dir))

try:
    import utils
except ImportError:
    print(f"Error: Could not import 'utils'. Make sure utils.py is in '{module_dir}'")
    sys.exit(1)

def process_incremental_dataset(input_path, output_path, dataset_type="Train"):
    print(f"--- Processing {dataset_type} Data ---")
    print(f"Loading data from: {input_path}")
    
    try:
        # Load JSON. The Cohesentia format typically needs transposing 
        # because StoryIDs are columns.
        df = pd.read_json(input_path).T
    except ValueError as e:
        print(f"Error reading JSON file: {e}")
        return

    # List to collect rows before creating DataFrame (much faster than appending to DF)
    output_rows = []

    # Iterate through each story
    for index, row in df.iterrows():
        try:
            _title_raw = row['Title']
            _story_id = row['StoryID']
            _incremental_data = row['IncrementalData']
            
            _sentences = _incremental_data['sentences']
            _score = _incremental_data['consensus_score']
            _reasons_data = _incremental_data['reasons']

            # Use utils logic to map reasons to sentences
            # Note: utils.get_reasons returns a dict: { "sentence_text": [r1, r2... r7, score] }
            _sentence_reasons_obj = utils.get_reasons(_story_id, _reasons_data, _sentences, _score)
            
            # Preprocess Title once per story
            _title_processed = utils.preprocess_sentences(
                _title_raw, lower=True, remove_puntuation=True, remove_stopwords=True, lemmatize=True
            )

            if _title_processed == 'REMOVE':
                _title_processed = ""

            # --- NEW: Variable to hold the previous sentence for context ---
            _previous_sentence_processed = ""

            # Iterate through processed sentences for this story
            for sent_text, reasons_array in _sentence_reasons_obj.items():
                
                # Unpack reasons and score
                _r1, _r2, _r3, _r4, _r5, _r6, _r7, _target = reasons_array

                # Preprocess the sentence text
                _sentence_processed = utils.preprocess_sentences(
                    sent_text, lower=True, remove_puntuation=True, remove_stopwords=True, lemmatize=True
                )

                if _sentence_processed == 'REMOVE':
                    continue

                # --- NEW: Combine Title + Previous Sentence (if exists) + Current Sentence ---
                if _previous_sentence_processed:
                    # Format: Title - Previous Sentence - Current Sentence
                    combined_text = f"{_title_processed} - {_previous_sentence_processed} {_sentence_processed}".strip()
                else:
                    # First sentence case: Title - Current Sentence
                    combined_text = f"{_title_processed} - {_sentence_processed}".strip()

                # Collect the row
                output_rows.append({
                    "sentence": combined_text,
                    "score": _target,
                    "r1": _r1,
                    "r2": _r2,
                    "r3": _r3,
                    "r4": _r4,
                    "r5": _r5,
                    "r6": _r6,
                    "r7": _r7
                })

                # --- NEW: Update the previous sentence for the next iteration ---
                _previous_sentence_processed = _sentence_processed

        except KeyError as e:
            print(f"Skipping story {index} due to missing key: {e}")
            continue

    # Create DataFrame and Save
    if output_rows:
        columns = ["sentence", "score", "r1", "r2", "r3", "r4", "r5", "r6", "r7"]
        df_output = pd.DataFrame(output_rows, columns=columns)
        
        df_output.to_csv(output_path, index=False)
        print(f"Saved {len(df_output)} rows to {output_path}")
    else:
        print("No data processed.")

def main():
    parser = argparse.ArgumentParser(description="Create Incremental (Sentence-level) Dataset CSVs from JSON")
    
    # Default paths based on your example
    parser.add_argument("--train_input", type=str, default="../../data/cohesentia/TrainData.json", help="Path to TrainData.json")
    parser.add_argument("--test_input", type=str, default="../../data/cohesentia/TestData.json", help="Path to TestData.json")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save CSV outputs")
    
    args = parser.parse_args()

    # Process Train
    if os.path.exists(args.train_input):
        train_output = os.path.join(args.output_dir, "TrainDataIncremental.csv")
        process_incremental_dataset(args.train_input, train_output, "Train")
    else:
        print(f"Warning: Train input not found at {args.train_input}")

    # Process Test
    if os.path.exists(args.test_input):
        test_output = os.path.join(args.output_dir, "TestDataIncremental.csv")
        process_incremental_dataset(args.test_input, test_output, "Test")
    else:
        print(f"Warning: Test input not found at {args.test_input}")

if __name__ == "__main__":
    main()