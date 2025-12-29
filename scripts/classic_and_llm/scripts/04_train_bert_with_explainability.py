"""
=============================================================================
Script: 04_train_bert_with_explainability.py
Description: Train BERT-based classifier with best hyperparameters and generate
             comprehensive explainability reports using LIME, SHAP, 
             Integrated Gradients, and Attention Visualization
             
Model Name: BERT_[MODEL_VARIANT]_[TIMESTAMP]
=============================================================================
"""

import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
from datetime import datetime

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
    'output_dir': './output/bert_trained',
    
    # Model configuration
    'model_name': 'bert-base-uncased',  # Pretrained model to use
    
    # Training hyperparameters (update with your best params from search)
    'hyperparameters': {
        'dropout': 0.1,
        'num_train_epochs': 6,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'warmup_steps': 500,
        'gradient_accumulation_steps': 1,
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_epsilon': 1e-8,
    },
    
    # Explainability settings
    'explainability': {
        'enabled': True,
        'methods': ['lime', 'shap', 'integrated_gradients', 'attention'],
        'sample_indices': [0, 1, 2],  # Indices of samples to explain
        'lime_num_features': 15,
        'ig_steps': 50,  # Steps for Integrated Gradients
        'save_images': True,
    }
}


def run_bert_training_and_explainability():
    """Main function to train BERT and generate explainability reports."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_variant = CONFIG['model_name'].replace('/', '_').replace('-', '_')
    model_name_full = f"BERT_{model_variant}_{timestamp}"
    
    print("=" * 80)
    print(f"TRAINING BERT CLASSIFIER WITH EXPLAINABILITY")
    print(f"Model Name: {model_name_full}")
    print("=" * 80)
    
    # Create output directory
    output_dir = os.path.join(CONFIG['output_dir'], model_name_full)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize NLP Model
    nlp = NLP_Model(output_dir=output_dir)
    
    # Load and preprocess data
    print(f"\nLoading dataset from: {CONFIG['dataset_path']}")
    nlp.load_csv(CONFIG['dataset_path'], verbose=True)
    nlp.preprocess_text('text', lower=True)
    
    # Train BERT model
    print(f"\nTraining BERT model: {CONFIG['model_name']}")
    print(f"Hyperparameters: {CONFIG['hyperparameters']}")
    
    hp = CONFIG['hyperparameters']
    result = nlp.build_bert_classifier(
        model_name=CONFIG['model_name'],
        dropout=hp['dropout'],
        num_train_epochs=hp['num_train_epochs'],
        batch_size=hp['batch_size'],
        learning_rate=hp['learning_rate'],
        weight_decay=hp['weight_decay'],
        warmup_steps=hp['warmup_steps'],
        gradient_accumulation_steps=hp['gradient_accumulation_steps'],
        adam_beta1=hp['adam_beta1'],
        adam_beta2=hp['adam_beta2'],
        adam_epsilon=hp['adam_epsilon'],
        verbose=True,
        override=True
    )
    
    model = nlp.single_output_model
    tokenizer = nlp.vectorizer_model
    
    print("\n" + "=" * 60)
    print("BERT MODEL TRAINING COMPLETE")
    print("=" * 60)
    
    # ==========================================================================
    # EVALUATION ON TEST SET
    # ==========================================================================
    if CONFIG['test_dataset_path']:
        print(f"\nEvaluating on test dataset: {CONFIG['test_dataset_path']}")
        report, tp, tn, fp, fn = nlp.get_bert_classsifier_test_score(
            test_dataset_path=CONFIG['test_dataset_path'],
            model_name=CONFIG['model_name'],
            verbose=True
        )
        print("\nClassification Report:")
        print(report)
    
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
        explainer.tokenizer = tokenizer
        explainer.dataset = nlp.data.copy()
        explainer.class_names = ['Negative', 'Positive']  # Adjust based on your classes
        explainer.model_type = 'transformers'
        explainer.bow = False  # Use subword tokenization for transformers
        
        # Create explainability output directory
        explain_dir = os.path.join(output_dir, 'explanations')
        os.makedirs(explain_dir, exist_ok=True)
        
        methods = CONFIG['explainability']['methods']
        sample_indices = CONFIG['explainability']['sample_indices']
        
        for idx in sample_indices:
            if idx >= len(nlp.data):
                print(f"Warning: Sample index {idx} out of range, skipping...")
                continue
            
            sample = nlp.data.iloc[idx]
            print(f"\n{'='*60}")
            print(f"Explaining sample {idx}:")
            print(f"Text: {sample['text'][:100]}...")
            print(f"True Label: {sample['score']}")
            print("=" * 60)
            
            # -----------------------------------------------------------------
            # LIME Explanation
            # -----------------------------------------------------------------
            if 'lime' in methods:
                print(f"\n--- Generating LIME explanation ---")
                try:
                    output_path = os.path.join(explain_dir, f'lime_sample_{idx}') if CONFIG['explainability']['save_images'] else None
                    explainer.get_lime_explanation(
                        sample_idx=idx,
                        num_features=CONFIG['explainability']['lime_num_features'],
                        output_image_path=output_path,
                        verbose=True
                    )
                    print(f"LIME explanation saved to: {output_path}.png")
                except Exception as e:
                    print(f"Error generating LIME explanation: {e}")
            
            # -----------------------------------------------------------------
            # SHAP Explanation
            # -----------------------------------------------------------------
            if 'shap' in methods:
                print(f"\n--- Generating SHAP explanation ---")
                try:
                    output_path = os.path.join(explain_dir, f'shap_sample_{idx}') if CONFIG['explainability']['save_images'] else None
                    explainer.get_shap_explanation(
                        sample_idx=idx,
                        output_image_path=output_path,
                        verbose=True
                    )
                    print(f"SHAP explanation saved to: {output_path}.png")
                except Exception as e:
                    print(f"Error generating SHAP explanation: {e}")
            
            # -----------------------------------------------------------------
            # Integrated Gradients Explanation
            # -----------------------------------------------------------------
            if 'integrated_gradients' in methods:
                print(f"\n--- Generating Integrated Gradients explanation ---")
                try:
                    true_class = int(sample['score'])
                    target_class = 1 if true_class == 0 else 0  # Explain opposite class
                    
                    output_path = os.path.join(explain_dir, f'ig_sample_{idx}') if CONFIG['explainability']['save_images'] else None
                    explainer.integrated_gradients_explanation(
                        sample_idx=idx,
                        true_class=true_class,
                        target_class=target_class,
                        steps=CONFIG['explainability']['ig_steps'],
                        output_image_path=output_path,
                        verbose=True
                    )
                    print(f"Integrated Gradients explanation saved to: {output_path}.png")
                except Exception as e:
                    print(f"Error generating Integrated Gradients explanation: {e}")
            
            # -----------------------------------------------------------------
            # Attention Visualization
            # -----------------------------------------------------------------
            if 'attention' in methods:
                print(f"\n--- Generating Attention visualization ---")
                try:
                    output_path = os.path.join(explain_dir, f'attention_sample_{idx}') if CONFIG['explainability']['save_images'] else None
                    explainer.visualize_bert_attention(
                        sample_idx=idx,
                        output_image_path=output_path
                    )
                    print(f"Attention visualization saved to: {output_path}.png")
                except Exception as e:
                    print(f"Error generating Attention visualization: {e}")
        
        print(f"\nAll explanations saved to: {explain_dir}")
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(CONFIG, f)
    
    # Save hyperparameters as JSON for easy reading
    import json
    hp_path = os.path.join(output_dir, 'hyperparameters.json')
    with open(hp_path, 'w') as f:
        json.dump(CONFIG['hyperparameters'], f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING AND EXPLAINABILITY COMPLETE")
    print("=" * 80)
    print(f"\nModel Name: {model_name_full}")
    print(f"Output Directory: {output_dir}")
    print(f"\nFiles saved:")
    print(f"  - Model files (model-{CONFIG['model_name']}-{timestamp}/)")
    print(f"  - Tokenizer files (tokeninzer-{CONFIG['model_name']}-{timestamp}/)")
    print(f"  - config.pkl")
    print(f"  - hyperparameters.json")
    print(f"  - Classification report and confusion matrix")
    if CONFIG['explainability']['enabled']:
        print(f"  - Explanations directory with:")
        for method in CONFIG['explainability']['methods']:
            print(f"      - {method.upper()} visualizations")
    
    return {
        'model_name': model_name_full,
        'model': model,
        'tokenizer': tokenizer,
        'output_dir': output_dir,
        'hyperparameters': CONFIG['hyperparameters']
    }


def load_best_params_from_search(search_params_path):
    """Load best parameters from BERT hyperparameter search results."""
    with open(search_params_path, 'rb') as f:
        params = pickle.load(f)
    return params


if __name__ == "__main__":
    # Update the dataset path before running
    CONFIG['dataset_path'] = input("Enter the path to your training dataset CSV: ").strip()
    
    if not os.path.exists(CONFIG['dataset_path']):
        print(f"Error: Dataset not found at {CONFIG['dataset_path']}")
        sys.exit(1)
    
    # Optional test dataset
    test_path = input("Enter path to test dataset (or press Enter to skip): ").strip()
    if test_path and os.path.exists(test_path):
        CONFIG['test_dataset_path'] = test_path
    
    # Model selection
    print("\nAvailable BERT models:")
    print("  1. bert-base-uncased")
    print("  2. bert-large-uncased")
    print("  3. distilbert-base-uncased")
    print("  4. roberta-base")
    print("  5. albert-base-v2")
    print("  6. Enter custom model name")
    
    choice = input(f"\nSelect model (1-6, press Enter for {CONFIG['model_name']}): ").strip()
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
    
    # Optionally load best params from search
    load_params = input("\nLoad best params from hyperparameter search? (y/n): ").strip().lower()
    if load_params == 'y':
        search_path = input("Enter path to best_bert_params.pkl file: ").strip()
        if os.path.exists(search_path):
            try:
                best_params = load_best_params_from_search(search_path)
                # Map the Optuna params to our config format
                param_mapping = {
                    'learning_rate': 'learning_rate',
                    'batch_size': 'batch_size',
                    'dropout': 'dropout',
                    'weight_decay': 'weight_decay',
                    'warmup_steps': 'warmup_steps',
                    'gradient_accumulation_steps': 'gradient_accumulation_steps',
                    'adam_beta1': 'adam_beta1',
                    'adam_beta2': 'adam_beta2',
                    'adam_epsilon': 'adam_epsilon'
                }
                for optuna_key, config_key in param_mapping.items():
                    if optuna_key in best_params:
                        CONFIG['hyperparameters'][config_key] = best_params[optuna_key]
                print(f"Loaded best parameters: {best_params}")
            except Exception as e:
                print(f"Error loading parameters: {e}")
                print("Using default parameters instead.")
    
    # Explainability options
    enable_explain = input("\nEnable explainability? (y/n, press Enter for yes): ").strip().lower()
    if enable_explain == 'n':
        CONFIG['explainability']['enabled'] = False
    else:
        print("\nAvailable explainability methods:")
        print("  1. LIME")
        print("  2. SHAP")
        print("  3. Integrated Gradients")
        print("  4. Attention Visualization")
        print("  5. All methods (default)")
        
        method_choice = input("Select methods (comma-separated numbers, e.g., 1,2,3 or press Enter for all): ").strip()
        if method_choice:
            method_map = {
                '1': 'lime',
                '2': 'shap',
                '3': 'integrated_gradients',
                '4': 'attention'
            }
            selected = [method_map[m.strip()] for m in method_choice.split(',') if m.strip() in method_map]
            if selected:
                CONFIG['explainability']['methods'] = selected
    
    print(f"\nStarting BERT training with:")
    print(f"  Model: {CONFIG['model_name']}")
    print(f"  Dataset: {CONFIG['dataset_path']}")
    print(f"  Hyperparameters: {CONFIG['hyperparameters']}")
    print(f"  Explainability: {CONFIG['explainability']['enabled']}")
    if CONFIG['explainability']['enabled']:
        print(f"  Methods: {CONFIG['explainability']['methods']}")
    
    results = run_bert_training_and_explainability()
