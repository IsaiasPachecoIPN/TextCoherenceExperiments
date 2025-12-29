import pandas as pd
import numpy as np
import os
import argparse
import sys

def get_annot_reason_score(json_arr, sentence_idx):
    """
    Parses the annotation JSON to determine which reasons apply to a specific sentence.
    Returns a binary list of length 7 corresponding to R1-R7.
    """
    reason_array = [0] * 7
    
    # Iterate through annotators (annot0, annot1, etc.)
    for key, value in json_arr.items():
        if key.startswith('annot'):
            # Check if this annotator flagged this sentence index
            if str(sentence_idx) in value:
                reasons = value[str(sentence_idx)]
                for reason in reasons:
                    # Reason indices in JSON seem to be 1-based (1-8), we map 1-7 to 0-6
                    # The notebook logic ignores reason 8 (index 7)
                    reason_array_idx = reason - 1
                    if 0 <= reason_array_idx < 7:
                        reason_array[reason_array_idx] = 1
    return reason_array

def get_sentence_reason_text(reason_array, sentence):
    """
    Generates the text explanation for why a sentence is incoherent based on the reason array.
    """
    comment = ""
    reasons_counter = 0
    
    explanations = [
        f"-The sentence '{sentence}' doesn't make sense <r1>1</r1>",
        f"-The sentence '{sentence}' discusses an entity which has not been introduced yet <r2>1</r2>",
        f"-The relation between this sentence '{sentence}' and previous ones doesn’t make sense <r3>1</r3>",
        f"-The new sentence '{sentence}' contains information inconsistent with previous presented data <r4>1</r4>",
        f"-The new sentence '{sentence}' contains information inconsistent with the knowledge about the world <r5>1</r5>",
        f"-The new sentence '{sentence}' is not relevant to the title <r6>1</r6>",
        f"-The new sentence '{sentence}' is not relevant to previous data in the story <r7>1</r7>"
    ]
    
    conjunctions = [
        "", # Not used for first item
        " and discusses an entity which has not been introduced yet <r2>1</r2>",
        " and the relation between it and previous ones doesn’t make sense <r3>1</r3>",
        " and contains information inconsistent with previous presented data <r4>1</r4>",
        " and contains information inconsistent with the knowledge about the world <r5>1</r5>",
        " and is not relevant to the title <r6>1</r6>",
        " and is not relevant to previous data in the story <r7>1</r7>"
    ]

    for i in range(7):
        if reason_array[i] == 1:
            if reasons_counter > 0:
                comment += conjunctions[i]
            else:
                comment += explanations[i]
            reasons_counter += 1
            
    return comment

def generate_cot_comment(score, incoherent_sentences):
    """
    Generates the Chain-of-Thought (CoT) block based on the consensus score
    and the accumulated incoherent sentence details.
    """
    # Clean up formatting for the incoherent sentences block
    if not incoherent_sentences.strip():
        problematic_block = "\n\t- No problematic sentences"
    else:
        problematic_block = incoherent_sentences

    if score == 1:
        return f"""1. Overall Coherence:\n\t- The story is not coherent at all because it contains some problematic sentences\n2. Problematic sentences:{problematic_block}\n3. Conclusion\n\t - The story is not coherent at all"""
    elif score == 2:
        return f"""1. Overall Coherence:\n\t- The story is not coherent because it contains some problematic sentences\n2. Problematic sentences:{problematic_block}\n3. Conclusion\n\t - The story is not coherent"""
    elif score == 3:
        return f"""1. Overall Coherence:\n\t- The story is maybe yes, maybe not coherent because it contains some problematic sentences\n2. Problematic sentences:{problematic_block}\n3. Conclusion\n\t - The story is maybe coherent"""
    elif score == 4:
        return f"""1. Overall Coherence:\n\t- The story is coherent even when it contains some problematic sentences\n2. Problematic sentences:{problematic_block}\n3. Conclusion\n\t - The story is coherent"""
    elif score == 5:
        return f"""1. Overall Coherence:\n\t- The story is definitely coherent even when it contains some problematic sentences\n2. Problematic sentences:{problematic_block}\n3. Conclusion\n\t - The story is definitely coherent"""
    
    return "Error: Invalid Score"

def process_dataset(file_path):
    """
    Main logic to read JSON and convert to structured DataFrame with CoT.
    """
    print(f"Loading data from: {file_path}")
    try:
        data = pd.read_json(file_path)
        data = data.T
        print(data.head())
    except ValueError as e:
        print(f"Error reading JSON file: {e}")
        return None

    titles = []
    texts = []
    scores = []
    story_ids = []
    cot_comments = []
    
    # Store aggregate reasons for the whole story (logical OR of all sentences)
    # List of lists (N_stories x 7)
    story_reasons_matrix = [] 

    # Iterate over the raw dictionary returned by read_json (which might be column-oriented)
    # It is safer to iterate values if read_json returned a dict-like structure or rows if it's a DF
    
    # To align with the notebook logic which iterated `for row in cohesentia_test_data:`
    # where that variable was a DataFrame. The notebook treated the index as the row key.
    
    for index, row in data.iterrows():
        title = row['Title']
        text = row['Text']
        
        # Access nested JSON structures
        inc_data = row['IncrementalData']
        score = inc_data.get('consensus_score')
        story_id = row['StoryID']
        
        # Initialize variables for this story
        incoherent_sentences_text = ""
        sentences_list = inc_data['sentences']
        reasons_json = inc_data['reasons']
        
        # Accumulator for reasons for this specific story
        current_story_reasons = [0] * 7 

        # Analyze sentences
        for idx, sentence in enumerate(sentences_list):
            # Get binary reasons for this sentence
            sentence_reasons = get_annot_reason_score(reasons_json, idx)
            
            # Update story-level reasons (Logical OR)
            for r_i in range(7):
                if sentence_reasons[r_i] == 1:
                    current_story_reasons[r_i] = 1

            # If there are reasons, generate textual explanation
            if 1 in sentence_reasons:
                reason_text = get_sentence_reason_text(sentence_reasons, sentence)
                incoherent_sentences_text += "\n\t" + reason_text

        # Generate CoT
        cot_block = generate_cot_comment(score, incoherent_sentences_text)

        # Append to lists
        titles.append(title)
        texts.append(f"Title: {title} Text: {text}")
        scores.append(score)
        story_ids.append(story_id)
        cot_comments.append(cot_block)
        story_reasons_matrix.append(current_story_reasons)

    # Create Initial DataFrame
    df = pd.DataFrame({
        '[ID]': story_ids,
        'text': texts,
        '[score]': scores,
        '[COT]': cot_comments
    })

    # Add Reason Columns [R1] to [R7]
    reason_cols = pd.DataFrame(story_reasons_matrix, columns=[f'[R{i}]' for i in range(1, 8)])
    final_df = pd.concat([df, reason_cols], axis=1)

    return final_df

def balance_class_5(df, target_sample_size=64):
    """
    Downsamples rows with score 5 to match a target size (class balancing).
    """
    print(f"Balancing dataset: Downsampling Score 5 to {target_sample_size} samples...")
    
    df_score_5 = df[df['[score]'] == 5]
    df_others = df[df['[score]'] != 5]

    if len(df_score_5) > target_sample_size:
        df_score_5_balanced = df_score_5.sample(n=target_sample_size, random_state=42)
    else:
        df_score_5_balanced = df_score_5
    
    balanced_df = pd.concat([df_others, df_score_5_balanced], ignore_index=True)
    
    # Sort by ID just to keep things tidy, though not strictly necessary
    balanced_df = balanced_df.sort_values(by='[ID]')
    
    print("New score distribution:")
    print(balanced_df['[score]'].value_counts().sort_index())
    
    return balanced_df

def main():
    parser = argparse.ArgumentParser(description="Convert Cohesentia JSON datasets to CoT CSV format.")
    parser.add_argument("--train_input", type=str, default="./data/TrainData.json", help="Path to input Train JSON")
    parser.add_argument("--test_input", type=str, default="./data/TestData.json", help="Path to input Test JSON")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save CSV outputs")
    
    args = parser.parse_args()

    # 1. Process Test Data (No Balancing)
    if os.path.exists(args.test_input):
        print("--- Processing Test Data ---")
        test_df = process_dataset(args.test_input)
        if test_df is not None:
            output_path = os.path.join(args.output_dir, "cohesentia_test_cot.csv")
            test_df.to_csv(output_path, index=False)
            print(f"Test data saved to: {output_path}")
            print(f"Total rows: {len(test_df)}")
    else:
        print(f"Warning: Test input file not found at {args.test_input}")

    # 2. Process Train Data (With Balancing)
    if os.path.exists(args.train_input):
        print("\n--- Processing Train Data ---")
        train_df = process_dataset(args.train_input)
        if train_df is not None:
            # Apply balancing specifically for the training set as per notebook logic
            balanced_train_df = balance_class_5(train_df)
            
            output_path = os.path.join(args.output_dir, "cohesentia_train_cot_balanced.csv")
            balanced_train_df.to_csv(output_path, index=False)
            print(f"Train data saved to: {output_path}")
            print(f"Total rows: {len(balanced_train_df)}")
    else:
        print(f"Warning: Train input file not found at {args.train_input}")

if __name__ == "__main__":
    main()