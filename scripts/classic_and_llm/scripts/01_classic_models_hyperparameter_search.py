"""
=============================================================================
Script: 01_classic_models_hyperparameter_search.py
Description: Hyperparameter search for classic NLP models using Optuna
             Supports: Logistic Regression, SVM, Random Forest, XGBoost, 
                       LightGBM, Naive Bayes, KNN, Decision Tree, MLP
=============================================================================
"""

import os
import sys
import pickle
import optuna
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clasic_models import NLP_Model

# =============================================================================
# CONFIGURATION - Modify these parameters as needed
# =============================================================================
CONFIG = {
    'dataset_path': 'path/to/your/dataset.csv',  # UPDATE THIS PATH
    'output_dir': './output/classic_hp_search',
    'embedding_name': 'tfidf',  # Options: 'word_count', 'tfidf', 'fasttext', 'word2vec', 'glove'
    'ngram_range': (1, 2),
    'n_trials': 100,
    'metric': 'f1',  # Options: 'accuracy', 'f1', 'precision'
    'cv_folds': 5,
    'random_state': 42,
    'models_to_tune': [
        'logistic_regression',
        'svm',
        'random_forest',
        'xgboost',
        'lightgbm',
        'mlp',
        'knn',
        'decision_tree',
        'sgd'
    ]
}


def get_model_and_params(trial, model_name, n_features):
    """
    Define hyperparameter search space for each model type.
    Returns a configured model instance.
    """
    
    if model_name == 'logistic_regression':
        solver = trial.suggest_categorical('lr_solver', ['lbfgs', 'liblinear', 'saga'])
        C = trial.suggest_float('lr_C', 1e-4, 100.0, log=True)
        
        if solver == 'liblinear':
            penalty = trial.suggest_categorical('lr_penalty_liblinear', ['l1', 'l2'])
        elif solver == 'saga':
            penalty = trial.suggest_categorical('lr_penalty_saga', ['l1', 'l2', 'elasticnet'])
        else:
            penalty = 'l2'
        
        params = {'solver': solver, 'C': C, 'penalty': penalty, 'max_iter': 2000, 
                  'random_state': CONFIG['random_state']}
        
        if solver == 'saga' and penalty == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('lr_l1_ratio', 0.0, 1.0)
            
        return LogisticRegression(**params)
    
    elif model_name == 'svm':
        kernel = trial.suggest_categorical('svm_kernel', ['linear', 'rbf', 'poly'])
        C = trial.suggest_float('svm_C', 1e-3, 100.0, log=True)
        
        params = {'C': C, 'kernel': kernel, 'probability': True, 
                  'random_state': CONFIG['random_state']}
        
        if kernel in ['rbf', 'poly']:
            params['gamma'] = trial.suggest_float('svm_gamma', 1e-4, 1.0, log=True)
        if kernel == 'poly':
            params['degree'] = trial.suggest_int('svm_degree', 2, 5)
            
        return SVC(**params)
    
    elif model_name == 'random_forest':
        return RandomForestClassifier(
            n_estimators=trial.suggest_int('rf_n_estimators', 50, 500),
            max_depth=trial.suggest_int('rf_max_depth', 3, 30),
            min_samples_split=trial.suggest_int('rf_min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('rf_min_samples_leaf', 1, 10),
            max_features=trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None]),
            n_jobs=-1,
            random_state=CONFIG['random_state']
        )
    
    elif model_name == 'xgboost':
        return XGBClassifier(
            n_estimators=trial.suggest_int('xgb_n_estimators', 50, 500),
            max_depth=trial.suggest_int('xgb_max_depth', 3, 15),
            learning_rate=trial.suggest_float('xgb_learning_rate', 1e-3, 0.3, log=True),
            subsample=trial.suggest_float('xgb_subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0),
            reg_alpha=trial.suggest_float('xgb_reg_alpha', 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float('xgb_reg_lambda', 1e-8, 10.0, log=True),
            n_jobs=-1,
            random_state=CONFIG['random_state'],
            use_label_encoder=False,
            eval_metric='logloss'
        )
    
    elif model_name == 'lightgbm':
        return LGBMClassifier(
            n_estimators=trial.suggest_int('lgbm_n_estimators', 50, 500),
            max_depth=trial.suggest_int('lgbm_max_depth', 3, 15),
            learning_rate=trial.suggest_float('lgbm_learning_rate', 1e-3, 0.3, log=True),
            num_leaves=trial.suggest_int('lgbm_num_leaves', 20, 150),
            subsample=trial.suggest_float('lgbm_subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('lgbm_colsample_bytree', 0.5, 1.0),
            reg_alpha=trial.suggest_float('lgbm_reg_alpha', 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float('lgbm_reg_lambda', 1e-8, 10.0, log=True),
            n_jobs=-1,
            random_state=CONFIG['random_state'],
            verbose=-1
        )
    
    elif model_name == 'mlp':
        n_layers = trial.suggest_int('mlp_n_layers', 1, 4)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'mlp_layer_{i}_units', 32, 256))
        
        return MLPClassifier(
            hidden_layer_sizes=tuple(layers),
            activation=trial.suggest_categorical('mlp_activation', ['relu', 'tanh']),
            alpha=trial.suggest_float('mlp_alpha', 1e-5, 1e-2, log=True),
            learning_rate_init=trial.suggest_float('mlp_learning_rate', 1e-4, 1e-2, log=True),
            max_iter=1000,
            early_stopping=True,
            random_state=CONFIG['random_state']
        )
    
    elif model_name == 'knn':
        return KNeighborsClassifier(
            n_neighbors=trial.suggest_int('knn_n_neighbors', 3, 30),
            weights=trial.suggest_categorical('knn_weights', ['uniform', 'distance']),
            metric=trial.suggest_categorical('knn_metric', ['euclidean', 'manhattan', 'cosine']),
            n_jobs=-1
        )
    
    elif model_name == 'decision_tree':
        return DecisionTreeClassifier(
            max_depth=trial.suggest_int('dt_max_depth', 3, 30),
            min_samples_split=trial.suggest_int('dt_min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('dt_min_samples_leaf', 1, 10),
            criterion=trial.suggest_categorical('dt_criterion', ['gini', 'entropy']),
            random_state=CONFIG['random_state']
        )
    
    elif model_name == 'sgd':
        loss = trial.suggest_categorical('sgd_loss', ['log_loss', 'hinge', 'modified_huber'])
        penalty = trial.suggest_categorical('sgd_penalty', ['l1', 'l2', 'elasticnet'])
        
        params = {
            'loss': loss,
            'penalty': penalty,
            'alpha': trial.suggest_float('sgd_alpha', 1e-6, 1e-1, log=True),
            'max_iter': 2000,
            'early_stopping': True,
            'n_jobs': -1,
            'random_state': CONFIG['random_state']
        }
        
        if penalty == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('sgd_l1_ratio', 0.0, 1.0)
            
        return SGDClassifier(**params)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_objective(X, y, model_name, n_features):
    """Create an objective function for Optuna optimization."""
    
    def objective(trial):
        try:
            model = get_model_and_params(trial, model_name, n_features)
            
            # Use stratified k-fold cross-validation
            cv = StratifiedKFold(n_splits=CONFIG['cv_folds'], shuffle=True, 
                                 random_state=CONFIG['random_state'])
            
            # Get cross-validation scores
            if CONFIG['metric'] == 'accuracy':
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            elif CONFIG['metric'] == 'f1':
                scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
            elif CONFIG['metric'] == 'precision':
                scores = cross_val_score(model, X, y, cv=cv, scoring='precision_weighted', n_jobs=-1)
            else:
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            
            return scores.mean()
            
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return float('-inf')
    
    return objective


def run_hyperparameter_search():
    """Main function to run hyperparameter search for all specified models."""
    
    print("=" * 80)
    print("CLASSIC NLP MODELS - HYPERPARAMETER SEARCH")
    print("=" * 80)
    
    # Initialize NLP Model
    nlp = NLP_Model(output_dir=CONFIG['output_dir'])
    
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
    
    # Prepare data for training
    X = nlp.word_count_dataset.drop(columns=['[SCORE]'])
    y = nlp.word_count_dataset['[SCORE]']
    n_features = X.shape[1]
    
    print(f"\nData shape: {X.shape}")
    print(f"Number of features: {n_features}")
    
    # Store all results
    all_results = {}
    
    # Run optimization for each model
    for model_name in CONFIG['models_to_tune']:
        print(f"\n{'=' * 60}")
        print(f"Tuning: {model_name.upper()}")
        print(f"{'=' * 60}")
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=CONFIG['random_state']),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )
        
        # Create objective function
        objective = create_objective(X, y, model_name, n_features)
        
        # Run optimization
        study.optimize(objective, n_trials=CONFIG['n_trials'], show_progress_bar=True)
        
        # Store results
        all_results[model_name] = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
        
        print(f"\nBest {CONFIG['metric'].upper()} for {model_name}: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        
        # Save study
        study_path = os.path.join(CONFIG['output_dir'], f'study_{model_name}.pkl')
        os.makedirs(os.path.dirname(study_path), exist_ok=True)
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
    
    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(CONFIG['output_dir'], f'all_results_{timestamp}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    
    # Print summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE - SUMMARY")
    print("=" * 80)
    
    # Find best overall model
    best_model = max(all_results, key=lambda x: all_results[x]['best_score'])
    
    print(f"\n{'Model':<25} {'Best Score':<15} {'N Trials':<10}")
    print("-" * 50)
    for model_name, result in sorted(all_results.items(), key=lambda x: x[1]['best_score'], reverse=True):
        print(f"{model_name:<25} {result['best_score']:.4f}          {result['n_trials']}")
    
    print(f"\n🏆 Best Overall Model: {best_model.upper()}")
    print(f"   Score: {all_results[best_model]['best_score']:.4f}")
    print(f"   Parameters: {all_results[best_model]['best_params']}")
    print(f"\nResults saved to: {results_path}")
    
    return all_results


if __name__ == "__main__":
    # Update the dataset path before running
    CONFIG['dataset_path'] = input("Enter the path to your dataset CSV: ").strip()
    
    if not os.path.exists(CONFIG['dataset_path']):
        print(f"Error: Dataset not found at {CONFIG['dataset_path']}")
        sys.exit(1)
    
    results = run_hyperparameter_search()
