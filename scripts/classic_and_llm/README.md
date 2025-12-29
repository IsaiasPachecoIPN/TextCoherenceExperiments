# NLP Model Library

This library (`clasic_models.py`) provides a comprehensive set of tools for Natural Language Processing (NLP) tasks, ranging from classical machine learning models to modern deep learning approaches like BERT. It simplifies the process of data loading, preprocessing, vectorization, model training, and evaluation.

## Features

- **Data Management**: Easy loading of CSV datasets and vocabulary building.
- **Preprocessing**: Text cleaning, lowercasing, and column merging.
- **Vectorization**: Support for multiple embedding techniques:
  - Word Count (Bag of Words)
  - TF-IDF
  - FastText
  - Word2Vec
  - GloVe
- **Classical ML Models**: Train and evaluate multiple scikit-learn models (Logistic Regression, SGD, etc.) with automatic cross-validation and best model selection.
- **Deep Learning (BERT)**: Full support for BERT-based classification, including hyperparameter optimization using Optuna.
- **Evaluation**: Detailed classification reports, confusion matrices, and visualization tools.

## Dependencies

Ensure you have the following Python packages installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn gensim transformers torch optuna imbalanced-learn coloredlogs lime shap selenium bertviz captum ipython
```

> **Note:** For saving explanations as images, this library uses **Selenium** with **ChromeDriver**. Ensure you have Google Chrome installed and `chromedriver` is in your system PATH, or provide the path to the executable when calling the methods.

## Dataset Format

The input CSV file must contain at least the following two columns:

- **`text`**: The input text data to be classified.
- **`score`**: The target label (integer) for classification.

**Example CSV structure:**

| text | score |
|------|-------|
| "This is a great product" | 1 |
| "I did not like this" | 0 |
| "Average experience" | 1 |

## Usage Examples

### 1. Initialization and Data Loading

```python
from clasicModel import NLP_Model

# Initialize the model wrapper with a custom output directory
nlp = NLP_Model(output_dir='./my_experiments')

# Load your dataset (must contain 'text' and 'score' columns for this library's conventions)
nlp.load_csv('path/to/your/dataset.csv', verbose=True)
```

### 2. Preprocessing

```python
# Preprocess the text column (remove special chars, optional lowercase)
nlp.preprocess_text('text', lower=True)

# Build vocabulary from the dataset
nlp.build_vocabulary(verbose=True)
```

### 3. Vectorization

Choose one of the available vectorization methods before training.

**Option A: TF-IDF**
```python
nlp.build_tfidf_vectorizer(ngram_range=(1, 2))
```

**Option B: Word2Vec**
```python
nlp.build_word2vect_vectorizer()
```

### 4. Training Classical Models

**Using Default Models:**
The library comes with a set of default models (Logistic Regression, SGDClassifier variants).

```python
# Train using the vectorizer built in the previous step
nlp.build_singleOutputClassifier(embedding_name='tfidf', metric='f1')
```

**Using Custom Models:**
You can pass your own list of scikit-learn compatible classifiers.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

custom_models = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    SVC(kernel='linear')
]

nlp.build_singleOutputClassifier(
    models=custom_models,
    embedding_name='tfidf',
    metric='accuracy'
)
```

### 5. Training BERT Classifier

**Hyperparameter Tuning (Optional):**
Use Optuna to find the best hyperparameters.

```python
best_params = nlp.explore_bert_classifier_hyper_parameters(
    model_name='bert-base-uncased',
    n_trials=20
)
```

**Training:**
Train the BERT model with specific parameters.

```python
nlp.build_bert_classifier(
    model_name='bert-base-uncased',
    num_train_epochs=4,
    batch_size=16,
    learning_rate=2e-5
)
```

### 6. Evaluation

Evaluate the trained model on a separate test dataset.

**For Classical Models:**
```python
report, tp, tn, fp, fn = nlp.get_classifier_test_score(
    test_dataset_path='path/to/test_dataset.csv',
    embedding_name='tfidf'
)
print(report)
```

**For BERT Models:**
```python
report, tp, tn, fp, fn = nlp.get_bert_classsifier_test_score(
    test_dataset_path='path/to/test_dataset.csv',
    model_name='bert-base-uncased'
)
print(report)
```

## Output Structure

The library automatically creates an output directory (default: `./output`) containing:
- Pickled datasets and vectorizers.
- Saved models (`.pkl` for classical, directory structure for BERT).
- Logs and visualization figures.
- Classification reports and confusion matrices.

# Model Explainer Library

This library (`model_explainer.py`) provides tools to interpret and visualize the predictions of your NLP models. It supports various explanation methods including LIME, SHAP, Integrated Gradients, and Attention Visualization.

## Features

- **LIME**: Local Interpretable Model-agnostic Explanations.
- **SHAP**: SHapley Additive exPlanations.
- **Integrated Gradients**: Feature attribution for deep learning models (Captum).
- **BERT Attention**: Visualization of attention heads (BertViz).
- **Image Export**: Save explanations as images using Selenium.

## Usage Examples

### 1. Initialization

You need to instantiate the `ModelExplainer` and manually set the model, tokenizer, and dataset.

```python
from model_explainer import ModelExplainer

# Initialize
explainer = ModelExplainer()

# Set required attributes
explainer.model = my_trained_model       # Your trained model object
explainer.tokenizer = my_tokenizer       # Tokenizer (e.g., from transformers or sklearn)
explainer.dataset = my_dataframe         # DataFrame with 'text' and 'score' columns
explainer.class_names = ['Negative', 'Positive']
explainer.model_type = 'transformers'    # Options: 'classic', 'transformers', 'fasttext', 'glove'
```

### 2. LIME Explanation

Explain a specific sample using LIME.

```python
explainer.get_lime_explanation(
    sample_idx=0,
    num_features=10,
    output_image_path='lime_output'
)
```

### 3. SHAP Explanation

Explain a prediction using SHAP values.

```python
explainer.get_shap_explanation(
    sample_idx=0,
    output_image_path='shap_output'
)
```

### 4. Integrated Gradients (BERT only)

Visualize feature importance using Integrated Gradients.

```python
explainer.integrated_gradients_explanation(
    sample_idx=0,
    true_class=1,
    target_class=0,
    steps=50,
    output_image_path='ig_output'
)
```

### 5. BERT Attention Visualization

Visualize the attention mechanism of a BERT model.

```python
explainer.visualize_bert_attention(
    sample_idx=0,
    output_image_path='attention_output'
)
```

---

# Training Scripts

The `scripts/` directory contains ready-to-use scripts for hyperparameter search, model training, and explainability analysis. Each script includes the model name in its outputs for easy identification.

## Available Scripts

| Script | Description |
|--------|-------------|
| `01_classic_models_hyperparameter_search.py` | Hyperparameter search for classic ML models using Optuna |
| `02_bert_hyperparameter_search.py` | Hyperparameter search for BERT models using Optuna |
| `03_train_classic_model_with_explainability.py` | Train classic models with best params + LIME/SHAP explanations |
| `04_train_bert_with_explainability.py` | Train BERT with best params + full explainability suite |
| `05_full_pipeline.py` | Complete end-to-end pipeline for both model types |

## Script Details

### 1. Classic Models Hyperparameter Search (`01_classic_models_hyperparameter_search.py`)

Performs hyperparameter tuning for multiple classic ML models using Optuna's Bayesian optimization.

**Supported Models:**
- Logistic Regression
- SVM (Support Vector Machine)
- Random Forest
- XGBoost
- LightGBM
- MLP (Multi-Layer Perceptron)
- K-Nearest Neighbors
- Decision Tree
- SGD Classifier

**Usage:**
```bash
python scripts/01_classic_models_hyperparameter_search.py
```

**Configuration:**
Edit the `CONFIG` dictionary in the script to customize:
- `dataset_path`: Path to your CSV dataset
- `embedding_name`: 'tfidf', 'word_count', 'fasttext', 'word2vec', 'glove'
- `n_trials`: Number of Optuna trials per model
- `models_to_tune`: List of models to optimize

### 2. BERT Hyperparameter Search (`02_bert_hyperparameter_search.py`)

Performs hyperparameter tuning for BERT-based transformers using Optuna.

**Optimized Parameters:**
- Learning rate
- Batch size
- Dropout
- Weight decay
- Warmup steps
- Gradient accumulation steps
- Adam optimizer parameters (beta1, beta2, epsilon)

**Usage:**
```bash
python scripts/02_bert_hyperparameter_search.py
```

**Supported BERT Models:**
- bert-base-uncased
- bert-large-uncased
- distilbert-base-uncased
- roberta-base
- albert-base-v2
- Any HuggingFace transformer model

### 3. Train Classic Model with Explainability (`03_train_classic_model_with_explainability.py`)

Trains a classic ML model with specified hyperparameters and generates LIME and SHAP explanations.

**Model Name Format:** `{model_type}_{embedding}_{timestamp}`

**Features:**
- Load best parameters from hyperparameter search
- Train model with cross-validation
- Generate classification report and confusion matrix
- Create LIME explanations for specified samples
- Create SHAP explanations for specified samples
- Save all outputs with model name in the path

**Usage:**
```bash
python scripts/03_train_classic_model_with_explainability.py
```

### 4. Train BERT with Explainability (`04_train_bert_with_explainability.py`)

Trains a BERT model with specified hyperparameters and generates comprehensive explanations.

**Model Name Format:** `BERT_{model_variant}_{timestamp}`

**Explainability Methods:**
- **LIME**: Feature importance through perturbation
- **SHAP**: SHapley Additive exPlanations
- **Integrated Gradients**: Gradient-based feature attribution
- **Attention Visualization**: BERT attention head visualization

**Usage:**
```bash
python scripts/04_train_bert_with_explainability.py
```

### 5. Full Pipeline (`05_full_pipeline.py`)

A comprehensive pipeline that combines all steps: hyperparameter search, training, and explainability.

**Modes:**
- `full`: Complete pipeline (HP search → Training → Explainability)
- `hp_search_only`: Only run hyperparameter search
- `train_only`: Only train with default/loaded parameters

**Command Line Usage:**
```bash
# BERT full pipeline
python scripts/05_full_pipeline.py --train data/train.csv --test data/test.csv --model_type bert --mode full

# Classic models with TF-IDF
python scripts/05_full_pipeline.py --train data/train.csv --model_type classic --embedding tfidf --n_trials 50

# HP search only
python scripts/05_full_pipeline.py --train data/train.csv --model_type bert --mode hp_search_only
```

**Interactive Usage:**
```bash
python scripts/05_full_pipeline.py
# Follow the prompts to configure the pipeline
```

## Output Structure

Each script creates an organized output directory:

```
output/
├── classic_hp_search/
│   ├── study_logistic_regression.pkl
│   ├── study_svm.pkl
│   └── all_results_{timestamp}.pkl
│
├── bert_hp_search/
│   ├── best_bert_model/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── optuna_study.pkl
│   └── best_bert_params.pkl
│
├── classic_trained/
│   └── logistic_regression_tfidf_{timestamp}/
│       ├── model.pkl
│       ├── vectorizer.pkl
│       ├── config.pkl
│       ├── {model_name}_report/
│       │   ├── report.csv
│       │   ├── hyperparams.csv
│       │   ├── full_report.txt
│       │   └── confusion_matrix.png
│       └── explanations/
│           ├── lime_sample_0.png
│           ├── shap_sample_0.png
│           └── ...
│
└── bert_trained/
    └── BERT_bert_base_uncased_{timestamp}/
        ├── model-bert-base-uncased-{timestamp}/
        ├── tokeninzer-bert-base-uncased-{timestamp}/
        ├── config.pkl
        ├── hyperparameters.json
        └── explanations/
            ├── lime_sample_0.png
            ├── shap_sample_0.png
            ├── ig_sample_0.png
            └── attention_sample_0.png
```

## Workflow Examples

### Example 1: Complete Classic Model Workflow

```bash
# Step 1: Run hyperparameter search
python scripts/01_classic_models_hyperparameter_search.py

# Step 2: Train best model with explainability
python scripts/03_train_classic_model_with_explainability.py
# When prompted, load the best parameters from step 1
```

### Example 2: Complete BERT Workflow

```bash
# Step 1: Run hyperparameter search
python scripts/02_bert_hyperparameter_search.py

# Step 2: Train with best parameters and generate explanations
python scripts/04_train_bert_with_explainability.py
# When prompted, load the best parameters from step 1
```

### Example 3: Quick Full Pipeline

```bash
# Run everything in one command
python scripts/05_full_pipeline.py --train data/train.csv --model_type bert --mode full --n_trials 30
```

## Tips

1. **GPU Acceleration**: For BERT training, ensure you have CUDA installed for GPU acceleration.

2. **Memory Management**: For large datasets with BERT, reduce `batch_size` if you encounter OOM errors.

3. **Hyperparameter Search Time**: Start with fewer trials (10-20) for initial exploration, then increase for production.

4. **Explainability Output**: Set `save_images=True` in config to save explanation visualizations as PNG files.

5. **Continue Search**: Use `continue_searching=True` to resume an interrupted hyperparameter search.

```
