# Cohesentia Chain-of-Thought (CoT) Dataset Creator

This project contains a Python script used to convert the raw JSON Cohesentia datasets (Train and Test) into formatted CSV files suitable for Fine-Tuning Large Language Models (LLMs) using Chain-of-Thought reasoning.

## Overview

The script `create_cot_dataset.py` processes raw story data annotated with coherence scores. It generates a "Chain of Thought" text block for each story that explains *why* a story received a specific score by highlighting incoherent sentences and specific reasoning tags (R1 through R7).

### Features
1.  **JSON Parsing**: Reads complex nested JSON structures containing story text, sentence lists, and annotator reasons.
2.  **CoT Generation**: Constructs a structured reasoning block (`[COT]`) summarizing the coherence of the story.
3.  **Reason Mapping**: Flags specific sentences with XML-style tags (e.g., `<r1>1</r1>`) inside the CoT explanation.
4.  **Class Balancing**: Automatically balances the training dataset by downsampling stories with a perfect score (5) to prevent model bias, matching the logic used in the original research notebook.
5.  **Data Columns**: Generates binary columns `[R1]` through `[R7]` indicating the presence of specific error types in the story.

## Example 

```sh
python create_cot_dataset.py --train_input "/data/cohesentia/TrainData.json" --test_input "/data/cohesentia/TestData.json" --output_dir .
```

## Requirements

* Python 3.x
* pandas
* numpy

To install dependencies:

```bash
pip install pandas numpy