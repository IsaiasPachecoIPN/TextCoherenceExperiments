"""
=============================================================================
Script: 03_train_classic_model_with_explainability.py
Description: Train classic NLP models with best hyperparameters and generate
             explainability reports using LIME and SHAP
             
Model Name: [MODEL_NAME]_[EMBEDDING]_[TIMESTAMP]
=============================================================================
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clasic_models import NLP_Model
from model_explainer import ModelExplainer

# =============================================================================
# CONFIGURATION - Modify these parameters as needed
# =============================================================================
CONFIG = {
    'dataset_path': 'path/to/your/dataset.csv',  # UPDATE THIS PATH
    'test_dataset_path': None,  # Optional: separate test dataset path
    'output_dir': './output/classic_trained',
    'embedding_name': 'tfidf',  # Options: 'word_count', 'tfidf', 'fasttext', 'word2vec', 'glove'
    'ngram_range': (1, 2),
    'random_state': 42,
    'test_size': 0.2,
    
    # Model selection - choose one
    'model_type': 'logistic_regression',  # Options: logistic_regression, svm, random_forest, 
                                           # xgboost, lightgbm, mlp, knn, decision_tree, sgd
    
    # Hyperparameters for each model type (update with your best params from search)
    'model_params': {
        'logistic_regression': {
            'solver': 'lbfgs',
            'C': 1.0,
            'penalty': 'l2',
            'max_iter': 2000,
        },
        'svm': {
            'C': 10.0,
            'kernel': 'rbf',
            'gamma': 0.01,
            'probability': True,  # Required for explainability
        },
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'n_jobs': -1,
        },
        'xgboost': {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
        },
        'lightgbm': {
            'n_estimators': 200,
            'max_depth': 10,
            'learning_rate': 0.1,
            'num_leaves': 50,
            'verbose': -1,
        },
        'mlp': {
            'hidden_layer_sizes': (128, 64),
            'activation': 'relu',
            'alpha': 0.0001,
            'learning_rate_init': 0.001,
            'max_iter': 1000,
            'early_stopping': True,
        },
        'knn': {
            'n_neighbors': 5,
            'weights': 'distance',
            'metric': 'cosine',
            'n_jobs': -1,
        },
        'decision_tree': {
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'criterion': 'gini',
        },
        'sgd': {
            'loss': 'log_loss',
            'penalty': 'l2',
            'alpha': 0.0001,
            'max_iter': 2000,
            'early_stopping': True,
        }
    },
    
    # Explainability settings
    'explainability': {
        'enabled': True,
        'lime_num_features': 15,
        'sample_indices': [0, 1, 2],  # Indices of samples to explain
        'save_images': True,
    }
}


def get_model(model_type, params, random_state):
    """Create a model instance based on model type and parameters."""
    
    all_params = {**params, 'random_state': random_state}
    
    if model_type == 'logistic_regression':
        return LogisticRegression(**all_params)
    elif model_type == 'svm':
        return SVC(**all_params)
    elif model_type == 'random_forest':
        return RandomForestClassifier(**all_params)
    elif model_type == 'xgboost':
        xgb_params = {k: v for k, v in all_params.items() if k != 'random_state'}
        xgb_params['random_state'] = random_state
        return XGBClassifier(**xgb_params)
    elif model_type == 'lightgbm':
        lgbm_params = {k: v for k, v in all_params.items() if k != 'random_state'}
        lgbm_params['random_state'] = random_state
        return LGBMClassifier(**lgbm_params)
    elif model_type == 'mlp':
        return MLPClassifier(**all_params)
    elif model_type == 'knn':
        knn_params = {k: v for k, v in params.items()}  # KNN doesn't use random_state
        return KNeighborsClassifier(**knn_params)
    elif model_type == 'decision_tree':
        return DecisionTreeClassifier(**all_params)
    elif model_type == 'sgd':
        sgd_params = {k: v for k, v in all_params.items()}
        return SGDClassifier(**sgd_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_training_and_explainability():
    """Main function to train model and generate explainability reports."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{CONFIG['model_type']}_{CONFIG['embedding_name']}_{timestamp}"
    
    print("=" * 80)
    print(f"TRAINING CLASSIC NLP MODEL WITH EXPLAINABILITY")
    print(f"Model Name: {model_name}")
    print("=" * 80)
    
    # Create output directory
    output_dir = os.path.join(CONFIG['output_dir'], model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize NLP Model
    nlp = NLP_Model(output_dir=output_dir)
    
    # Load and preprocess data
    print(f"\nLoading dataset from: {CONFIG['dataset_path']}")
    nlp.load_csv(CONFIG['dataset_path'], verbose=True)
    nlp.preprocess_text('text', lower=True)
    nlp.build_vocabulary(verbose=False)
    
    # Build vectorizer based on embedding type
    print(f"\nBuilding {CONFIG['embedding_name']} vectorizer...")
    if CONFIG['embedding_name'] == 'word_count':
        nlp.build_word_count_vectorizer(ngram_range=CONFIG['ngram_range'])
    elif CONFIG['embedding_name'] == 'tfidf':
        nlp.build_tfidf_vectorizer(ngram_range=CONFIG['ngram_range'])
    elif CONFIG['embedding_name'] == 'fasttext':
        nlp.build_fasttext_vectorizer()
    elif CONFIG['embedding_name'] == 'word2vec':
        nlp.build_word2vect_vectorizer()
    elif CONFIG['embedding_name'] == 'glove':
        nlp.build_glove_vectorizer()
    
    # Prepare data
    X = nlp.word_count_dataset.drop(columns=['[SCORE]'])
    y = nlp.word_count_dataset['[SCORE]']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'], stratify=y
    )
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Get model
    model_params = CONFIG['model_params'][CONFIG['model_type']]
    model = get_model(CONFIG['model_type'], model_params, CONFIG['random_state'])
    
    # Train model
    print(f"\nTraining {CONFIG['model_type']}...")
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    model_path = os.path.join(output_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")
    
    # Save vectorizer
    vectorizer_path = os.path.join(output_dir, 'vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(nlp.vectorizer_model, f)
    print(f"Vectorizer saved to: {vectorizer_path}")
    
    # Store in nlp object for later use
    nlp.single_output_model = model
    
    # Generate classification report
    nlp.generate_classification_report(y_test, y_pred, model_params, model_name, output_dir)
    
    # ==========================================================================
    # EXPLAINABILITY SECTION
    # ==========================================================================
    if CONFIG['explainability']['enabled']:
        print("\n" + "=" * 80)
        print("GENERATING EXPLAINABILITY REPORTS")
        print("=" * 80)
        
        # Initialize explainer
        explainer = ModelExplainer()
        explainer.model = model
        explainer.tokenizer = nlp.vectorizer_model
        explainer.dataset = nlp.data.copy()
        explainer.class_names = ['Negative', 'Positive']  # Adjust based on your classes
        
        # Set model type based on embedding
        if CONFIG['embedding_name'] in ['word_count', 'tfidf']:
            explainer.model_type = 'classic'
        else:
            explainer.model_type = CONFIG['embedding_name']
        
        # Create explainability output directory
        explain_dir = os.path.join(output_dir, 'explanations')
        os.makedirs(explain_dir, exist_ok=True)
        
        # Generate explanations for specified samples
        for idx in CONFIG['explainability']['sample_indices']:
            if idx >= len(nlp.data):
                print(f"Warning: Sample index {idx} out of range, skipping...")
                continue
                
            print(f"\n--- Generating LIME explanation for sample {idx} ---")
            
            try:
                output_path = os.path.join(explain_dir, f'lime_sample_{idx}') if CONFIG['explainability']['save_images'] else None
                explainer.get_lime_explanation(
                    sample_idx=idx,
                    num_features=CONFIG['explainability']['lime_num_features'],
                    output_image_path=output_path,
                    verbose=True
                )
            except Exception as e:
                print(f"Error generating LIME explanation for sample {idx}: {e}")
            
            print(f"\n--- Generating SHAP explanation for sample {idx} ---")
            
            try:
                output_path = os.path.join(explain_dir, f'shap_sample_{idx}') if CONFIG['explainability']['save_images'] else None
                explainer.get_shap_explanation(
                    sample_idx=idx,
                    output_image_path=output_path,
                    verbose=True
                )
            except Exception as e:
                print(f"Error generating SHAP explanation for sample {idx}: {e}")
        
        print(f"\nExplanations saved to: {explain_dir}")
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(CONFIG, f)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nModel Name: {model_name}")
    print(f"Output Directory: {output_dir}")
    print(f"\nFiles saved:")
    print(f"  - model.pkl")
    print(f"  - vectorizer.pkl")
    print(f"  - config.pkl")
    print(f"  - Classification report and confusion matrix")
    if CONFIG['explainability']['enabled']:
        print(f"  - Explanations (LIME & SHAP)")
    
    return {
        'model_name': model_name,
        'model': model,
        'vectorizer': nlp.vectorizer_model,
        'metrics': {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        },
        'output_dir': output_dir
    }


def load_best_params_from_search(search_results_path, model_type):
    """Load best parameters from hyperparameter search results."""
    with open(search_results_path, 'rb') as f:
        results = pickle.load(f)
    
    if model_type in results:
        return results[model_type]['best_params']
    else:
        raise ValueError(f"Model type {model_type} not found in search results")


if __name__ == "__main__":
    # Update the dataset path before running
    CONFIG['dataset_path'] = input("Enter the path to your dataset CSV: ").strip()
    
    if not os.path.exists(CONFIG['dataset_path']):
        print(f"Error: Dataset not found at {CONFIG['dataset_path']}")
        sys.exit(1)
    
    # Model selection
    print("\nAvailable models:")
    print("  1. logistic_regression")
    print("  2. svm")
    print("  3. random_forest")
    print("  4. xgboost")
    print("  5. lightgbm")
    print("  6. mlp")
    print("  7. knn")
    print("  8. decision_tree")
    print("  9. sgd")
    
    model_choice = input(f"\nSelect model (1-9, press Enter for {CONFIG['model_type']}): ").strip()
    model_map = {
        '1': 'logistic_regression',
        '2': 'svm',
        '3': 'random_forest',
        '4': 'xgboost',
        '5': 'lightgbm',
        '6': 'mlp',
        '7': 'knn',
        '8': 'decision_tree',
        '9': 'sgd'
    }
    
    if model_choice in model_map:
        CONFIG['model_type'] = model_map[model_choice]
    
    # Embedding selection
    print("\nAvailable embeddings:")
    print("  1. tfidf")
    print("  2. word_count")
    print("  3. fasttext")
    print("  4. word2vec")
    print("  5. glove")
    
    emb_choice = input(f"\nSelect embedding (1-5, press Enter for {CONFIG['embedding_name']}): ").strip()
    emb_map = {
        '1': 'tfidf',
        '2': 'word_count',
        '3': 'fasttext',
        '4': 'word2vec',
        '5': 'glove'
    }
    
    if emb_choice in emb_map:
        CONFIG['embedding_name'] = emb_map[emb_choice]
    
    # Optionally load best params from search
    load_params = input("\nLoad best params from hyperparameter search? (y/n): ").strip().lower()
    if load_params == 'y':
        search_path = input("Enter path to search results .pkl file: ").strip()
        if os.path.exists(search_path):
            try:
                best_params = load_best_params_from_search(search_path, CONFIG['model_type'])
                CONFIG['model_params'][CONFIG['model_type']].update(best_params)
                print(f"Loaded best parameters: {best_params}")
            except Exception as e:
                print(f"Error loading parameters: {e}")
                print("Using default parameters instead.")
    
    print(f"\nStarting training with:")
    print(f"  Model: {CONFIG['model_type']}")
    print(f"  Embedding: {CONFIG['embedding_name']}")
    print(f"  Dataset: {CONFIG['dataset_path']}")
    
    results = run_training_and_explainability()
