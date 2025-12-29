# Cohesentia Holistic Dataset Creator

This script, `dataset_holistic_creation.py`, transforms the raw Cohesentia JSON data into a **Binary Classification Format** (Holistic).

While the data is still processed at the sentence level (Title + Sentences), the target variable is simplified into a binary label (0 or 1) based on the overall consensus score of the story. Stories with ambiguous scores (score = 3) are strictly removed.

## Features

* **Preprocessing**: Lemmatizes and cleans both Title and Sentences using `spacy` (via shared `utils`).
* **Feature Combination**: Concatenates `{Title} {Sentence}` into a single input column.
* **Binary Labeling**: Converts the 1-5 scale into binary classes:
  * **Class 1 (Coherent)**: Score > 3 (i.e., 4 or 5).
  * **Class 0 (Incoherent)**: Score < 3 (i.e., 1 or 2).
  * **Discarded**: Score = 3.

## Requirements

* Python 3.x
* pandas
* spacy
* **Shared Utils**: Must have `utils.py` and `stopwords.txt` in `../utils/`.

### Directory Structure Assumption

```text
/root/
├── data
│   └── cohesentia
│       ├── README.json
│       ├── TestData.json
│       └── TrainData.json
├── preprocessing
│   ├── create_cot_dataset
│   │   ├── create_cot_dataset.py
│   │   └── README.md
│   ├── create_holistic_dataset
│   │   ├── dataset_holistic_creation.py
│   │   ├── README.md
│   │   ├── TestDataHolistic.csv
│   │   └── TrainDataHolistic.csv
│   ├── create_incremental_dataset
│   │   ├── dataset_incremental_creation.py
│   │   └── README.md
│   └── utils
│       ├── __pycache__
│       │   └── utils.cpython-311.pyc
│       ├── stopwords.txt
│       ├── stopwords.txt:Zone.Identifier
│       └── utils.py
└── scripts
    └── llm
        ├── optuna_search_hyperparams
        │   ├── main.py
        │   ├── optuna_search.py:Zone.Identifier
        │   └── README.md
        ├── train
        │   ├── main.py
        │   └── README.md
        └── utils
            ├── llm_utils_compute_metrics.py:Zone.Identifier
            ├── llm_utils.py
            ├── llm_utils.py:Zone.Identifier
            ├── README.md
            ├── reproducibility.py
            └── reproducibility.py:Zone.Identifier