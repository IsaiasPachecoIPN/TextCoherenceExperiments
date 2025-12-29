"""
=============================================================================
Script: 05_full_pipeline.py
Description: Complete end-to-end pipeline for NLP model training
             - Hyperparameter search (Classic or BERT)
             - Model training with best parameters
             - Explainability analysis
             - Model comparison and evaluation
             
Model Name: [MODEL_TYPE]_[EMBEDDING/BERT]_[TIMESTAMP]
=============================================================================
"""

import os
import sys
import pickle
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================
PIPELINE_CONFIG = {
    # Data paths
    'train_dataset_path': 'path/to/your/train_dataset.csv',
    'test_dataset_path': 'path/to/your/test_dataset.csv',  # Optional
    'output_base_dir': './output/pipeline',
    
    # Pipeline mode
    'mode': 'full',  # Options: 'hp_search_only', 'train_only', 'full'
    
    # Model type
    'model_type': 'bert',  # Options: 'classic', 'bert'
    
    # Classic model settings (if model_type == 'classic')
    'classic_config': {
        'embedding_name': 'tfidf',
        'ngram_range': (1, 2),
        'models_to_tune': ['logistic_regression', 'svm', 'random_forest', 'xgboost'],
        'n_trials': 50,
        'cv_folds': 5,
        'metric': 'f1'
    },
    
    # BERT settings (if model_type == 'bert')
    'bert_config': {
        'model_name': 'bert-base-uncased',
        'n_trials': 30
    },
    
    # Explainability
    'explainability': {
        'enabled': True,
        'methods': ['lime', 'shap'],  # For classic: ['lime', 'shap']
                                       # For BERT: ['lime', 'shap', 'integrated_gradients', 'attention']
        'sample_indices': [0, 1, 2, 3, 4],
        'lime_num_features': 15,
        'save_images': True
    },
    
    'random_state': 42
}


def run_classic_pipeline(config):
    """Run the complete pipeline for classic NLP models."""
    
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    import optuna
    from optuna.samplers import TPESampler
    
    from clasic_models import NLP_Model
    from model_explainer import ModelExplainer
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_name = f"classic_pipeline_{config['classic_config']['embedding_name']}_{timestamp}"
    output_dir = os.path.join(config['output_base_dir'], pipeline_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"CLASSIC NLP PIPELINE")
    print(f"Pipeline Name: {pipeline_name}")
    print("=" * 80)
    
    # Initialize NLP Model
    nlp = NLP_Model(output_dir=output_dir)
    
    # Load and preprocess data
    print(f"\nStep 1: Loading and preprocessing data...")
    nlp.load_csv(config['train_dataset_path'], verbose=True)
    nlp.preprocess_text('text', lower=True)
    nlp.build_vocabulary(verbose=False)
    
    # Build vectorizer
    print(f"\nStep 2: Building {config['classic_config']['embedding_name']} vectorizer...")
    emb = config['classic_config']['embedding_name']
    if emb == 'word_count':
        nlp.build_word_count_vectorizer(ngram_range=config['classic_config']['ngram_range'])
    elif emb == 'tfidf':
        nlp.build_tfidf_vectorizer(ngram_range=config['classic_config']['ngram_range'])
    elif emb == 'fasttext':
        nlp.build_fasttext_vectorizer()
    elif emb == 'word2vec':
        nlp.build_word2vect_vectorizer()
    elif emb == 'glove':
        nlp.build_glove_vectorizer()
    
    X = nlp.word_count_dataset.drop(columns=['[SCORE]'])
    y = nlp.word_count_dataset['[SCORE]']
    
    # Hyperparameter search
    if config['mode'] in ['hp_search_only', 'full']:
        print(f"\nStep 3: Running hyperparameter search...")
        
        all_results = {}
        
        for model_name in config['classic_config']['models_to_tune']:
            print(f"\n  Tuning: {model_name}")
            
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=config['random_state'])
            )
            
            def create_objective(model_name):
                def objective(trial):
                    if model_name == 'logistic_regression':
                        model = LogisticRegression(
                            C=trial.suggest_float('C', 1e-4, 100, log=True),
                            solver='lbfgs',
                            max_iter=2000,
                            random_state=config['random_state']
                        )
                    elif model_name == 'svm':
                        model = SVC(
                            C=trial.suggest_float('C', 1e-3, 100, log=True),
                            kernel=trial.suggest_categorical('kernel', ['linear', 'rbf']),
                            probability=True,
                            random_state=config['random_state']
                        )
                    elif model_name == 'random_forest':
                        model = RandomForestClassifier(
                            n_estimators=trial.suggest_int('n_estimators', 50, 300),
                            max_depth=trial.suggest_int('max_depth', 3, 20),
                            random_state=config['random_state'],
                            n_jobs=-1
                        )
                    elif model_name == 'xgboost':
                        model = XGBClassifier(
                            n_estimators=trial.suggest_int('n_estimators', 50, 300),
                            max_depth=trial.suggest_int('max_depth', 3, 12),
                            learning_rate=trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                            random_state=config['random_state'],
                            use_label_encoder=False,
                            eval_metric='logloss'
                        )
                    else:
                        model = LogisticRegression(random_state=config['random_state'])
                    
                    cv = StratifiedKFold(n_splits=config['classic_config']['cv_folds'], 
                                        shuffle=True, random_state=config['random_state'])
                    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
                    return scores.mean()
                
                return objective
            
            study.optimize(create_objective(model_name), 
                          n_trials=config['classic_config']['n_trials'], 
                          show_progress_bar=True)
            
            all_results[model_name] = {
                'best_params': study.best_params,
                'best_score': study.best_value
            }
            
            print(f"    Best {config['classic_config']['metric']}: {study.best_value:.4f}")
        
        # Save search results
        with open(os.path.join(output_dir, 'hp_search_results.pkl'), 'wb') as f:
            pickle.dump(all_results, f)
        
        # Find best model
        best_model_name = max(all_results, key=lambda x: all_results[x]['best_score'])
        best_params = all_results[best_model_name]['best_params']
        
        print(f"\n  Best Model: {best_model_name}")
        print(f"  Best Params: {best_params}")
    
    # Training
    if config['mode'] in ['train_only', 'full']:
        print(f"\nStep 4: Training best model...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config['random_state'], stratify=y
        )
        
        # Create best model
        if best_model_name == 'logistic_regression':
            model = LogisticRegression(**best_params, max_iter=2000, random_state=config['random_state'])
        elif best_model_name == 'svm':
            model = SVC(**best_params, probability=True, random_state=config['random_state'])
        elif best_model_name == 'random_forest':
            model = RandomForestClassifier(**best_params, random_state=config['random_state'], n_jobs=-1)
        elif best_model_name == 'xgboost':
            model = XGBClassifier(**best_params, random_state=config['random_state'], 
                                 use_label_encoder=False, eval_metric='logloss')
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted')
        }
        
        print(f"\n  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        
        # Save model
        with open(os.path.join(output_dir, f'{best_model_name}_model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        
        with open(os.path.join(output_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(nlp.vectorizer_model, f)
        
        nlp.single_output_model = model
        nlp.generate_classification_report(y_test, y_pred, best_params, best_model_name, output_dir)
    
    # Explainability
    if config['explainability']['enabled'] and config['mode'] in ['train_only', 'full']:
        print(f"\nStep 5: Generating explainability reports...")
        
        explainer = ModelExplainer()
        explainer.model = model
        explainer.tokenizer = nlp.vectorizer_model
        explainer.dataset = nlp.data.copy()
        explainer.class_names = ['Negative', 'Positive']
        
        if emb in ['word_count', 'tfidf']:
            explainer.model_type = 'classic'
        else:
            explainer.model_type = emb
        
        explain_dir = os.path.join(output_dir, 'explanations')
        os.makedirs(explain_dir, exist_ok=True)
        
        for idx in config['explainability']['sample_indices']:
            if idx >= len(nlp.data):
                continue
            
            for method in config['explainability']['methods']:
                output_path = os.path.join(explain_dir, f'{method}_sample_{idx}') if config['explainability']['save_images'] else None
                
                try:
                    if method == 'lime':
                        explainer.get_lime_explanation(idx, config['explainability']['lime_num_features'], output_path, verbose=False)
                    elif method == 'shap':
                        explainer.get_shap_explanation(idx, output_image_path=output_path, verbose=False)
                except Exception as e:
                    print(f"    Warning: Failed to generate {method} for sample {idx}: {e}")
        
        print(f"  Explanations saved to: {explain_dir}")
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    
    return {
        'pipeline_name': pipeline_name,
        'best_model': best_model_name,
        'best_params': best_params,
        'metrics': metrics if config['mode'] in ['train_only', 'full'] else None,
        'output_dir': output_dir
    }


def run_bert_pipeline(config):
    """Run the complete pipeline for BERT models."""
    
    from clasic_models import NLP_Model
    from model_explainer import ModelExplainer
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_variant = config['bert_config']['model_name'].replace('/', '_').replace('-', '_')
    pipeline_name = f"bert_pipeline_{model_variant}_{timestamp}"
    output_dir = os.path.join(config['output_base_dir'], pipeline_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"BERT PIPELINE")
    print(f"Pipeline Name: {pipeline_name}")
    print("=" * 80)
    
    # Initialize NLP Model
    nlp = NLP_Model(output_dir=output_dir)
    
    # Load and preprocess data
    print(f"\nStep 1: Loading and preprocessing data...")
    nlp.load_csv(config['train_dataset_path'], verbose=True)
    nlp.preprocess_text('text', lower=True)
    
    best_params = None
    
    # Hyperparameter search
    if config['mode'] in ['hp_search_only', 'full']:
        print(f"\nStep 2: Running BERT hyperparameter search...")
        
        best_params = nlp.explore_bert_classifier_hyper_parameters(
            model_name=config['bert_config']['model_name'],
            n_trials=config['bert_config']['n_trials'],
            verbose=True
        )
        
        with open(os.path.join(output_dir, 'best_bert_params.pkl'), 'wb') as f:
            pickle.dump(best_params, f)
        
        print(f"\n  Best Parameters: {best_params}")
    
    # Training with best params
    if config['mode'] in ['train_only', 'full']:
        print(f"\nStep 3: Training BERT with best parameters...")
        
        if best_params is None:
            # Use defaults
            best_params = {
                'learning_rate': 2e-5,
                'batch_size': 16,
                'dropout': 0.1,
                'weight_decay': 0.01,
                'warmup_steps': 500,
                'gradient_accumulation_steps': 1,
                'adam_beta1': 0.9,
                'adam_beta2': 0.999,
                'adam_epsilon': 1e-8
            }
        
        result = nlp.build_bert_classifier(
            model_name=config['bert_config']['model_name'],
            dropout=best_params.get('dropout', 0.1),
            num_train_epochs=6,
            batch_size=best_params.get('batch_size', 16),
            learning_rate=best_params.get('learning_rate', 2e-5),
            weight_decay=best_params.get('weight_decay', 0.01),
            warmup_steps=best_params.get('warmup_steps', 500),
            gradient_accumulation_steps=best_params.get('gradient_accumulation_steps', 1),
            adam_beta1=best_params.get('adam_beta1', 0.9),
            adam_beta2=best_params.get('adam_beta2', 0.999),
            adam_epsilon=best_params.get('adam_epsilon', 1e-8),
            verbose=True,
            override=True
        )
    
    # Explainability
    if config['explainability']['enabled'] and config['mode'] in ['train_only', 'full']:
        print(f"\nStep 4: Generating explainability reports...")
        
        explainer = ModelExplainer()
        explainer.model = nlp.single_output_model
        explainer.tokenizer = nlp.vectorizer_model
        explainer.dataset = nlp.data.copy()
        explainer.class_names = ['Negative', 'Positive']
        explainer.model_type = 'transformers'
        explainer.bow = False
        
        explain_dir = os.path.join(output_dir, 'explanations')
        os.makedirs(explain_dir, exist_ok=True)
        
        for idx in config['explainability']['sample_indices']:
            if idx >= len(nlp.data):
                continue
            
            for method in config['explainability']['methods']:
                output_path = os.path.join(explain_dir, f'{method}_sample_{idx}') if config['explainability']['save_images'] else None
                
                try:
                    if method == 'lime':
                        explainer.get_lime_explanation(idx, config['explainability']['lime_num_features'], output_path, verbose=False)
                    elif method == 'shap':
                        explainer.get_shap_explanation(idx, output_image_path=output_path, verbose=False)
                    elif method == 'integrated_gradients':
                        sample = nlp.data.iloc[idx]
                        explainer.integrated_gradients_explanation(idx, true_class=int(sample['score']), 
                                                                   output_image_path=output_path, verbose=False)
                    elif method == 'attention':
                        explainer.visualize_bert_attention(idx, output_image_path=output_path)
                except Exception as e:
                    print(f"    Warning: Failed to generate {method} for sample {idx}: {e}")
        
        print(f"  Explanations saved to: {explain_dir}")
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    
    return {
        'pipeline_name': pipeline_name,
        'best_params': best_params,
        'output_dir': output_dir
    }


def main():
    """Main entry point for the pipeline."""
    
    parser = argparse.ArgumentParser(description='NLP Model Training Pipeline')
    parser.add_argument('--train', type=str, required=True, help='Path to training dataset CSV')
    parser.add_argument('--test', type=str, help='Path to test dataset CSV (optional)')
    parser.add_argument('--output', type=str, default='./output/pipeline', help='Output directory')
    parser.add_argument('--model_type', type=str, choices=['classic', 'bert'], default='bert')
    parser.add_argument('--mode', type=str, choices=['hp_search_only', 'train_only', 'full'], default='full')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased')
    parser.add_argument('--embedding', type=str, default='tfidf', 
                       choices=['tfidf', 'word_count', 'fasttext', 'word2vec', 'glove'])
    parser.add_argument('--n_trials', type=int, default=30)
    parser.add_argument('--no_explain', action='store_true', help='Disable explainability')
    
    args = parser.parse_args()
    
    # Update config
    PIPELINE_CONFIG['train_dataset_path'] = args.train
    PIPELINE_CONFIG['test_dataset_path'] = args.test
    PIPELINE_CONFIG['output_base_dir'] = args.output
    PIPELINE_CONFIG['model_type'] = args.model_type
    PIPELINE_CONFIG['mode'] = args.mode
    PIPELINE_CONFIG['bert_config']['model_name'] = args.bert_model
    PIPELINE_CONFIG['bert_config']['n_trials'] = args.n_trials
    PIPELINE_CONFIG['classic_config']['embedding_name'] = args.embedding
    PIPELINE_CONFIG['classic_config']['n_trials'] = args.n_trials
    PIPELINE_CONFIG['explainability']['enabled'] = not args.no_explain
    
    if args.model_type == 'bert':
        PIPELINE_CONFIG['explainability']['methods'] = ['lime', 'shap', 'integrated_gradients', 'attention']
    else:
        PIPELINE_CONFIG['explainability']['methods'] = ['lime', 'shap']
    
    # Validate
    if not os.path.exists(args.train):
        print(f"Error: Training dataset not found at {args.train}")
        sys.exit(1)
    
    # Run pipeline
    if args.model_type == 'classic':
        results = run_classic_pipeline(PIPELINE_CONFIG)
    else:
        results = run_bert_pipeline(PIPELINE_CONFIG)
    
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode
        main()
    else:
        # Interactive mode
        print("=" * 80)
        print("NLP MODEL TRAINING PIPELINE - INTERACTIVE MODE")
        print("=" * 80)
        
        PIPELINE_CONFIG['train_dataset_path'] = input("\nEnter path to training dataset CSV: ").strip()
        
        if not os.path.exists(PIPELINE_CONFIG['train_dataset_path']):
            print(f"Error: Dataset not found at {PIPELINE_CONFIG['train_dataset_path']}")
            sys.exit(1)
        
        test_path = input("Enter path to test dataset (or press Enter to skip): ").strip()
        if test_path and os.path.exists(test_path):
            PIPELINE_CONFIG['test_dataset_path'] = test_path
        
        print("\nSelect model type:")
        print("  1. Classic (Logistic Regression, SVM, Random Forest, etc.)")
        print("  2. BERT (Transformer-based)")
        model_choice = input("Choice (1 or 2): ").strip()
        PIPELINE_CONFIG['model_type'] = 'bert' if model_choice == '2' else 'classic'
        
        print("\nSelect pipeline mode:")
        print("  1. Full (HP search + Training + Explainability)")
        print("  2. HP Search Only")
        print("  3. Train Only (use default/loaded params)")
        mode_choice = input("Choice (1, 2, or 3): ").strip()
        PIPELINE_CONFIG['mode'] = {
            '1': 'full',
            '2': 'hp_search_only',
            '3': 'train_only'
        }.get(mode_choice, 'full')
        
        n_trials = input(f"\nNumber of trials (press Enter for {PIPELINE_CONFIG['bert_config']['n_trials']}): ").strip()
        if n_trials:
            PIPELINE_CONFIG['bert_config']['n_trials'] = int(n_trials)
            PIPELINE_CONFIG['classic_config']['n_trials'] = int(n_trials)
        
        if PIPELINE_CONFIG['model_type'] == 'classic':
            PIPELINE_CONFIG['explainability']['methods'] = ['lime', 'shap']
            results = run_classic_pipeline(PIPELINE_CONFIG)
        else:
            PIPELINE_CONFIG['explainability']['methods'] = ['lime', 'shap', 'integrated_gradients', 'attention']
            results = run_bert_pipeline(PIPELINE_CONFIG)
