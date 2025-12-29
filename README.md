# Text Coherence Experiments

This repository contains a collection of scripts and datasets for conducting experiments on text coherence. It provides tools for data preprocessing, model training (classic ML and LLMs), hyperparameter optimization, and model explainability.

## Repository Structure

```
├── data
│   ├── cohesentia
│   │   ├── README.json
│   │   ├── TestData.json
│   │   └── TrainData.json
│   └── GCDC
│       ├── Clinton_test.csv
│       ├── Clinton_train.csv
│       ├── Enron_test.csv
│       ├── Enron_train.csv
│       ├── README.txt
│       ├── Yahoo_test.csv
│       ├── Yahoo_train.csv
│       ├── Yelp_test.csv
│       └── Yelp_train.csv
├── preprocessing
│   ├── create_cot_dataset
│   │   ├── create_cot_dataset.py
│   │   └── README.md
│   ├── create_generated_dataset
│   │   ├── main.py
│   │   └── README.md
│   ├── create_holistic_dataset
│   │   ├── dataset_holistic_creation.py
│   │   ├── README.md
│   │   ├── TestDataHolistic.csv
│   │   └── TrainDataHolistic.csv
│   ├── create_incremental_dataset
│   │   ├── dataset_incremental_creation.py
│   │   └── README.md
│   └── utils
│       ├── __pycache__
│       │   └── utils.cpython-311.pyc
│       ├── stopwords.txt
│       └── utils.py
└── scripts
    ├── classic_and_llm
    │   ├── clasic_models.py
    │   ├── model_explainer.py
    │   ├── README.md
    │   └── scripts
    │       ├── 01_classic_models_hyperparameter_search.py
    │       ├── 02_bert_hyperparameter_search.py
    │       ├── 03_train_classic_model_with_explainability.py
    │       ├── 04_train_bert_with_explainability.py
    │       └── 05_full_pipeline.py
    └── gen_llm
        ├── optuna_search_hyperparams
        │   ├── main.py
        │   └── README.md
        ├── train
        │   ├── main.py
        │   └── README.md
        └── utils
            ├── llm_utils.py
            ├── README.md
            └── reproducibility.py
```

## Data

This repository includes two main datasets for coherence analysis:

*   **Cohesentia**: A rich dataset in JSON format containing stories annotated with both holistic and incremental coherence scores. It also provides detailed reasons for incoherence at the sentence level.
*   **GCDC (Georgetown Coherence from Discovered Corpora)**: A collection of texts from various domains (Yahoo, Clinton emails, Enron emails, Yelp reviews) with coherence ratings from both expert annotators and crowd-sourced workers on Amazon Mechanical Turk.

For more details on each dataset, refer to the `README` files within their respective directories (`data/cohesentia/README.json` and `data/GCDC/README.txt`).

## Preprocessing

The `preprocessing` directory contains scripts to transform the raw datasets into formats suitable for different modeling approaches.

*   **`create_cot_dataset`**: Generates a dataset for fine-tuning Large Language Models (LLMs) using a Chain-of-Thought (CoT) approach. It creates detailed explanations for why a story has a particular coherence score.
*   **`create_generated_dataset`**: Contains scripts to generate synthetic stories using a base LLM, which can be used to augment the training data.
*   **`create_holistic_dataset`**: Transforms the Cohesentia dataset into a binary classification format, labeling each story as either "coherent" (1) or "incoherent" (0).
*   **`create_incremental_dataset`**: Creates a sentence-level dataset where the input for the model is `Title - Previous Sentence - Current Sentence`, allowing models to learn coherence in an incremental fashion.

## Scripts

The `scripts` directory is divided into two main parts, one for classic ML/BERT models and another for generative LLMs.

### Classic and LLM Models (`classic_and_llm`)

This section provides a comprehensive framework for training and explaining various NLP models.

*   **`clasic_models.py`**: A library for training and evaluating classic machine learning models (e.g., Logistic Regression, SVM) and BERT-based classifiers. It handles data loading, preprocessing, and vectorization (TF-IDF, Word2Vec, GloVe, FastText).
*   **`model_explainer.py`**: A tool for interpreting model predictions using methods like LIME, SHAP, Integrated Gradients, and BERT Attention Visualization.
*   **`scripts/`**: A collection of scripts that automate the entire pipeline:
    *   `01_classic_models_hyperparameter_search.py`: Finds the best hyperparameters for classic models using Optuna.
    *   `02_bert_hyperparameter_search.py`: Finds the best hyperparameters for BERT models using Optuna.
    *   `03_train_classic_model_with_explainability.py`: Trains a classic model with the best hyperparameters and generates explanations.
    *   `04_train_bert_with_explainability.py`: Trains a BERT model with the best hyperparameters and generates explanations.
    *   `05_full_pipeline.py`: An end-to-end script that combines hyperparameter search, training, and explainability for both model types.

### Generative LLM (`gen_llm`)

This section focuses on fine-tuning and evaluating generative Large Language Models for coherence tasks.

*   **`optuna_search_hyperparams`**: A script that uses Optuna and Unsloth to perform an efficient hyperparameter search for LoRA-based fine-tuning.
*   **`train`**: A script to perform the final production training run using the optimal hyperparameters discovered in the search phase.
*   **`utils/llm_utils.py`**: A powerful utility library that wraps around Unsloth, Hugging Face TRL, and Optuna to streamline the entire fine-tuning and evaluation workflow.

## How to Use

1.  **Explore the datasets** in the `data` directory to understand the available data.
2.  **Run the preprocessing scripts** in the `preprocessing` directory to prepare the data for your desired modeling approach.
3.  **Use the scripts** in the `scripts` directory to train, evaluate, and explain your models. Start with the hyperparameter search scripts to find the best configuration for your chosen model and dataset.

For detailed instructions on each step, please refer to the `README.md` files located within each subdirectory.
