# Cohesentia Incremental Dataset Creator

This project contains the `dataset_incremental_creation.py` script, designed to transform raw Cohesentia JSON datasets into an **Incremental Sentence-Level CSV format** suitable for training machine learning models on coherence tasks.

## Overview

This script processes story data by breaking it down into individual sentences. To provide context for coherence modeling, it constructs the input text by combining the **Story Title**, the **Previous Sentence** (if available), and the **Current Sentence**.

### Key Features

* **Incremental Context**: Each input row is formatted as: `Title - Previous Sentence - Current Sentence`. This allows the model to "see" what came before when evaluating the current sentence.
* **Granularity**: Operates at the sentence level rather than the full story level.
* **Preprocessing**: Uses `spacy` (via a shared `utils` module) to lower-case, remove punctuation, remove stopwords, and lemmatize text.
* **Reason Mapping**: Associates specific coherence error tags (`r1` through `r7`) with the exact sentence they apply to.
* **Dual Output**: Automatically processes both Training and Testing datasets if provided.

## Requirements

* Python 3.x
* pandas
* spacy
* **Spacy Model**: `en_core_web_sm`
* **Utils Module**: A `utils.py` file containing `get_reasons` and `preprocess_sentences` functions.
* **Stopwords**: A `stopwords.txt` file.

### Directory Structure Assumption

The script assumes the following directory layout by default:

```text
/root/
├── utils/
│   ├── utils.py
│   └── stopwords.txt
├── data/
│   └── cohesentia/
│       ├── TrainData.json
│       └── TestData.json
└── preprocessing/
    └── dataset_incremental_creation.py   <-- Run this script