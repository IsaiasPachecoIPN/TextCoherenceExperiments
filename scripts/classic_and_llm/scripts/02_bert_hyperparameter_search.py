"""
=============================================================================
Script: 02_bert_hyperparameter_search.py
Description: Hyperparameter search for BERT-based classifiers using Optuna
             Supports various pre-trained transformer models
=============================================================================
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clasic_models import NLP_Model

# =============================================================================
# CONFIGURATION - Modify these parameters as needed
# =============================================================================
CONFIG = {
    'dataset_path': 'path/to/your/dataset.csv',  # UPDATE THIS PATH
    'output_dir': './output/bert_hp_search',
    'model_name': 'bert-base-uncased',  # Options: 'bert-base-uncased', 'bert-large-uncased', 
                                         # 'distilbert-base-uncased', 'roberta-base', etc.
    'n_trials': 50,
    'continue_searching': False,  # Set to True to continue a previous search
    'study_name': None,  # Name for the Optuna study (None for default)
    'best_model_name': None,  # Name for saved model (None for default)
    
    # Initial parameters to try first (optional)
    'init_params': {
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
}


def run_bert_hyperparameter_search():
    """Main function to run BERT hyperparameter search using Optuna."""
    
    print("=" * 80)
    print("BERT CLASSIFIER - HYPERPARAMETER SEARCH")
    print("=" * 80)
    print(f"\nModel: {CONFIG['model_name']}")
    print(f"Number of trials: {CONFIG['n_trials']}")
    print(f"Continue from previous search: {CONFIG['continue_searching']}")
    
    # Initialize NLP Model
    nlp = NLP_Model(output_dir=CONFIG['output_dir'])
    
    # Load and preprocess data
    print(f"\nLoading dataset from: {CONFIG['dataset_path']}")
    nlp.load_csv(CONFIG['dataset_path'], verbose=True)
    nlp.preprocess_text('text', lower=True)
    
    # Run hyperparameter optimization
    print("\nStarting hyperparameter optimization...")
    print("This may take a while depending on dataset size and number of trials.\n")
    
    best_params = nlp.explore_bert_classifier_hyper_parameters(
        model_name=CONFIG['model_name'],
        n_trials=CONFIG['n_trials'],
        continue_searching=CONFIG['continue_searching'],
        best_model_name=CONFIG['best_model_name'],
        study_name=CONFIG['study_name'],
        init_params=CONFIG['init_params'],
        verbose=True
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 80)
    print(f"\nBest parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest model saved to: {CONFIG['output_dir']}/best_bert_model")
    print(f"Optuna study saved to: {CONFIG['output_dir']}/best_bert_model/optuna_study.pkl")
    
    # Save best params for easy loading
    import pickle
    params_path = os.path.join(CONFIG['output_dir'], 'best_bert_params.pkl')
    with open(params_path, 'wb') as f:
        pickle.dump(best_params, f)
    print(f"Best parameters saved to: {params_path}")
    
    return best_params


if __name__ == "__main__":
    # Update the dataset path before running
    CONFIG['dataset_path'] = input("Enter the path to your dataset CSV: ").strip()
    
    if not os.path.exists(CONFIG['dataset_path']):
        print(f"Error: Dataset not found at {CONFIG['dataset_path']}")
        sys.exit(1)
    
    # Optionally change model
    model_choice = input(f"Use {CONFIG['model_name']}? (y/n, press Enter for yes): ").strip().lower()
    if model_choice == 'n':
        print("\nAvailable models:")
        print("  1. bert-base-uncased")
        print("  2. bert-large-uncased")
        print("  3. distilbert-base-uncased")
        print("  4. roberta-base")
        print("  5. albert-base-v2")
        print("  6. Enter custom model name")
        
        choice = input("\nSelect model (1-6): ").strip()
        model_map = {
            '1': 'bert-base-uncased',
            '2': 'bert-large-uncased',
            '3': 'distilbert-base-uncased',
            '4': 'roberta-base',
            '5': 'albert-base-v2'
        }
        
        if choice in model_map:
            CONFIG['model_name'] = model_map[choice]
        elif choice == '6':
            CONFIG['model_name'] = input("Enter custom model name: ").strip()
    
    # Number of trials
    n_trials_input = input(f"Number of trials (press Enter for {CONFIG['n_trials']}): ").strip()
    if n_trials_input:
        CONFIG['n_trials'] = int(n_trials_input)
    
    print(f"\nStarting BERT hyperparameter search with:")
    print(f"  Dataset: {CONFIG['dataset_path']}")
    print(f"  Model: {CONFIG['model_name']}")
    print(f"  Trials: {CONFIG['n_trials']}")
    
    results = run_bert_hyperparameter_search()
