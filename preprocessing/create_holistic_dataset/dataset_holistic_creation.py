import pandas as pd
import sys
import os
import argparse

# Add the utils directory to path. 
module_dir = "../utils"
sys.path.append(os.path.abspath(module_dir))

try:
    import utils
except ImportError:
    print(f"Error: Could not import 'utils'. Make sure utils.py is in '{module_dir}'")
    sys.exit(1)

def process_holistic_dataset(input_path, output_path, dataset_type="Train"):
    print(f"--- Processing {dataset_type} Data (Holistic) ---")
    print(f"Loading data from: {input_path}")
    
    try:
        # Load JSON and transpose because Cohesentia keys are StoryIDs (columns)
        df = pd.read_json(input_path).T
    except ValueError as e:
        print(f"Error reading JSON file: {e}")
        return

    output_rows = []

    # Iterate through each story
    for index, row in df.iterrows():
        try:
            _title_raw = row['Title']
            # The `_incremental_data` variable is a dictionary containing information related to the
            # incremental data of a story. Within this dictionary, there are two key-value pairs being
            # accessed:
            _text_raw = row['Text']
            _holistic_data = row['HolisticData']
            
            # For Holistic, we process sentence by sentence but focus on the binary score mapping
            # The original logic seemed to map every single sentence to the story's overall binary score
            _consensus_score = _holistic_data['consensus_score']
            
            # Get Binary Score (0 or 1) using utils logic
            # Score > 3 -> 1 (Coherent)
            # Score < 3 -> 0 (Incoherent)
            # Score == 3 -> -1 (Discard)
            _binary_score = utils.get_score(_consensus_score)

            # If score is -1, we discard the entire story's sentences
            if _binary_score == -1:
                continue

            # Preprocess Title
            _title_processed = utils.preprocess_sentences(
                _title_raw, lower=True, remove_puntuation=True, remove_stopwords=True, lemmatize=True
            )
            if _title_processed == 'REMOVE':
                _title_processed = ""
                
            # Preprocess text
            _text_processed = utils.preprocess_sentences(
                _text_raw, lower=True, remove_puntuation=True, remove_stopwords=True, lemmatize=True
            )

            output_rows.append({
                "sentence":  _title_processed + " - " +_text_processed,
                "score": _binary_score
            })

        except KeyError as e:
            print(f"Skipping story {index} due to missing key: {e}")
            continue

    # Create DataFrame
    if output_rows:
        new_df = pd.DataFrame(output_rows)
        
        # Print stats for validation
        score_counts = new_df['score'].value_counts()
        print(f"Class Distribution for {dataset_type}:")
        print(score_counts)
        
        # Save
        new_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved {len(new_df)} rows to {output_path}")
    else:
        print("No valid data processed (check if all scores were 3).")

def main():
    parser = argparse.ArgumentParser(description="Create Holistic (Binary Classification) Dataset CSVs from JSON")
    
    # Default paths
    parser.add_argument("--train_input", type=str, default="../../data/cohesentia/TrainData.json", help="Path to TrainData.json")
    parser.add_argument("--test_input", type=str, default="../../data/cohesentia/TestData.json", help="Path to TestData.json")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save CSV outputs")
    
    args = parser.parse_args()

    # Process Train
    if os.path.exists(args.train_input):
        train_output = os.path.join(args.output_dir, "TrainDataHolistic.csv")
        process_holistic_dataset(args.train_input, train_output, "Train")
    else:
        print(f"Warning: Train input not found at {args.train_input}")

    # Process Test
    if os.path.exists(args.test_input):
        test_output = os.path.join(args.output_dir, "TestDataHolistic.csv")
        process_holistic_dataset(args.test_input, test_output, "Test")
    else:
        print(f"Warning: Test input not found at {args.test_input}")

if __name__ == "__main__":
    main()