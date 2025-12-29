
import torch
import json
import re
import traceback
import os
import wandb
import optuna
import pickle
import optunahub
import time 
import gc
import threading
import numpy                    as np
import pandas                   as pd
import matplotlib.pyplot        as plt
import seaborn                  as sns

from tqdm                       import tqdm
from IPython.display            import HTML, display

# from optuna.samplers            import AutoSampler          
from optuna.pruners             import HyperbandPruner   
from optuna.samplers            import TPESampler
from optuna.distributions       import IntDistribution, FloatDistribution, CategoricalDistribution
from optuna_dashboard           import run_server

from unsloth                    import FastLanguageModel
from unsloth                    import to_sharegpt
from unsloth                    import standardize_sharegpt
from unsloth                    import apply_chat_template
from unsloth                    import unsloth_train
from unsloth                    import is_bfloat16_supported
from unsloth                    import FastModel

from datasets                   import load_dataset
from datasets                   import Dataset
from datasets                   import DatasetDict
from datasets                   import concatenate_datasets

from torch.utils.data           import DataLoader

from sklearn.model_selection    import train_test_split
from sklearn.metrics            import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

from trl                        import SFTTrainer

from imblearn.under_sampling    import RandomUnderSampler
from transformers               import TrainingArguments
from transformers               import TrainerCallback
from transformers               import EarlyStoppingCallback
from transformers               import GenerationConfig
from transformers               import TextStreamer
from transformers               import StoppingCriteria, StoppingCriteriaList

from typing                     import Dict, List, Tuple
from collections                import Counter
from datetime                   import datetime

from reproducibility            import SEED 
from dotenv                     import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

print(f"Alert: Check if the ##Response is being generated correctly. If not, check the model name and the tokenizer.")

class ScoreStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.score_end_token = tokenizer.encode("[STOP]")[0]
        self.stop_count = 0  # Initialize a counter for the stop token

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the last generated token is the [STOP] token
        if input_ids[0][-1] == self.score_end_token:
            self.stop_count += 1
        
        # Stop generation after the second [STOP] token is detected
        return self.stop_count >= 2

class LLMModel:

    def __init__(self, model_name):

        self.model              = None
        self.tokenizer          = None
        self.model_name         = model_name
        self.train_dataset      = None
        self.test_dataset       = None

        #For trainer
        self.trainer_train_dataset = None
        self.trainer_eval_dataset  = None
        
        self.initial_params         = None
        self.is_optuna_search       = False
        
        # Configuration
        self.wandb_key              = os.getenv("WANDB_KEY")
        self.wandb_project          = os.getenv("WANDB_PROJECT", "Optuna-unsloth-Default")

        if not self.wandb_project:
            self.wandb_project = "Optuna-unsloth-Default"
        
        if not self.wandb_key:
            print("WARNING: WANDB_KEY not found in environment variables.")
            
        self.STUDY_PKL              = f"{self.wandb_project}_study.pkl"
        
    def send_notification(self, message):
        import requests
        if TOKEN is not None and CHAT_ID is not None:
            requests.get(f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}")
        else:
            pass
        

    def load_pretrained_model(self, max_seq_length=2048, load_in_4bit=True, dtype=None):
        try:
            if self.model_name == None:
                raise Exception("Model name is None")
            else:
                print(f"Loading model: {self.model_name}")

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = self.model_name,
                max_seq_length = int(max_seq_length),
                dtype = dtype,
                load_in_4bit = bool(load_in_4bit)
            )

        except Exception as e:
            print(e)
            print(traceback.format_exc())

        # print(f"Model: {self.model}")
        return self.model, self.tokenizer


    def generate_model_output(self, prompt , max_new_tokens=1024, temperature=1.0, top_k=50, top_p=1.0, do_sample=False, skip_special_tokens = True,  repetition_penalty=1.0, safety_margin=0.9):

        try:
            FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference

            # Set model to eval mode and disable gradients
            self.model.eval()

            inputs = self.tokenizer(prompt , return_tensors="pt").to(self.model.device)
            prompt_length = inputs["input_ids"].shape[1]

            # Calculate dynamic max_new_tokens
            max_available = int(self.model.config.max_position_embeddings * safety_margin)
            dynamic_max_new_tokens = min(
                max_new_tokens,  # User's original limit
                max_available - prompt_length,  # Hard cap based on remaining space
            )

            # Ensure at least 1 token can be generated
            dynamic_max_new_tokens = max(dynamic_max_new_tokens, 1)

            print(f"Prompt tokens: {prompt_length}/{max_available}")
            print(f"Generating up to {dynamic_max_new_tokens} new tokens")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=dynamic_max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty
                )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=skip_special_tokens)

        except Exception as e:
            print(e)

    def load_train_csv_dataset(self, file_path):

        try:

            if file_path == None:
                raise Exception("File path is None")

            dataset = pd.read_csv(file_path)
            if dataset.empty:
                raise Exception("Dataset is empty")

            self.train_dataset = dataset
            print(self.train_dataset.head())

        except Exception as e:
            print(e)

    def load_test_csv_dataset(self, file_path):

        try:

            if file_path == None:
                raise Exception("File path is None")

            dataset = pd.read_csv(file_path)
            if dataset.empty:
                raise Exception("Dataset is empty")

            self.test_dataset = dataset
            print(self.test_dataset.head())
        except Exception as e:
            print(e)


    def n_shot_experiment(self, prompt, max_new_tokens = 1024, n_shot=0, show_chart = False, save_path = None, verbose = False):

        try:

            greater_prompt_length = 0
            none_score_counter = 0

            # Extract n shots from the train dataset
            if n_shot > 0:
                class_0 = self.train_dataset[self.train_dataset['[score]'] == 0].sample(n_shot, random_state=42)
                class_1 = self.train_dataset[self.train_dataset['[score]'] == 1].sample(n_shot, random_state=42)

                print(f"Class 0: {class_0.shape}")
                print(f"Class 1: {class_1.shape}")

                n_shot_examples = pd.concat([class_0, class_1])

            y_true = []
            y_pred = []

            if show_chart:

                from IPython.display import clear_output, display

                plt.ion()
                fig, ax = plt.subplots(figsize=(12, 5))
                line_true, = ax.plot([], [], label="True Score", marker='o')
                line_pred, = ax.plot([], [], label="Predicted Score", marker='x')
                ax.set_title("Real-Time True vs Predicted Scores")
                ax.set_xlabel("Example Index")
                ax.set_ylabel("Score")
                ax.grid(True)
                ax.legend()

            for index, row in self.test_dataset.iterrows():
                print(f"Row: {index}")
                text = row['text']

                if n_shot > 0:
                    fs_prompt = self.build_few_shot_prompt(prompt, n_shot_examples, text)
                else:
                    fs_prompt = prompt.format(text, "")

                if len(fs_prompt) > greater_prompt_length:
                    greater_prompt_length = len(fs_prompt)
                    print(f"Greater prompt length: {greater_prompt_length}")

                gen_text = self.generate_model_output(fs_prompt, skip_special_tokens=False, max_new_tokens=max_new_tokens)

                print(f"Gen Text: {gen_text}")

                score = self.get_output_score(gen_text.split('Data Response:')[1])

                if verbose:
                    # print(f"Prompt: {fs_prompt}")
                    # print(f"Gen Text: {gen_text}")
                    print(f"*** Response generate: {gen_text.split('Data Response:')[1]} \n***")
                    print(f"*** Score extracted: {score} \n ***")

                if score is None:
                    print("************** Score is None")
                    none_score_counter += 1
                    # print(f"Text: {text}")
                    # print(f"Gen Text: {gen_text}")
                    continue

                y_true.append(row['[score]'])
                y_pred.append(score)

                if show_chart:
                    current_f1 = f1_score(y_true, y_pred, average='weighted')
                    print(f"F1 Score: {current_f1:.4f}")
                    ax.set_title(f"f1 weighted: {current_f1:.4f} - gpl {greater_prompt_length} - nsc: {none_score_counter}")
                    # Actualización del gráfico
                    line_true.set_data(range(len(y_pred)), y_true[:len(y_pred)])
                    line_pred.set_data(range(len(y_pred)), y_pred)
                    ax.relim()
                    ax.autoscale_view()
                    clear_output(wait=True)
                    display(fig)


            print("***" * 20)
            print(f"True: {y_true}")
            print(f"Pred: {y_pred}")
            print(f"F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
            print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
            print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
            print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
            print(classification_report(y_true, y_pred))

            self.get_cm_matrix(y_true, y_pred, n_shot=n_shot, save_path=save_path)

            if save_path is not None:
                with open(f"{save_path}.txt", 'w') as f:
                    f.write(f"True: {y_true}\n")
                    f.write(f"Pred: {y_pred}\n")
                    f.write(f"F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}\n")
                    f.write(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
                    f.write(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}\n")
                    f.write(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}\n")
                    f.write("\n")
                    f.write(f"Classification Report:\n")
                    f.write(classification_report(y_true, y_pred))
                    f.write("\n")
                    f.close()

            print("***" * 20)


        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())


    def get_output_score(self, text):

        """Extrae solo el valor del score de forma eficiente"""
        match = re.search(r"<score>\s*(\d+)\s*</score>", text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def get_cm_matrix(self, y_true, y_pred, n_shot=0, save_path = None):

        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión - {n_shot} shots')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        if save_path is not None:
            plt.savefig(f"{save_path}.png")
        plt.show()
        plt.close()

    def build_few_shot_prompt(self, prompt, n_shot_examples, text):

        prompt = prompt.split("## Input:")[0]
        prompt += "\n\n### EXAMPLE FORMAT - FOLLOW THIS STRUCTURE EXACTLY:\n"

        counter = 0
        for idx, example in n_shot_examples.iterrows():
            prompt += f"# Example: {counter + 1}:\n"
            prompt += f"# Data:\n{example['text']}\n"
            prompt += f"# Reasoning:\n<analysis_start>\n{example['[COT]']}\n<analysis_end>\n"
            prompt += f"# Response:\n<score>{example['[score]']}</score>\n"
            prompt += "\n# DO NOT FORMAT LIKE THIS:\n- General analysis without specific tags or structure\n- Missing sentence-by-sentence analysis\n- Score not in proper tag format\n\n"
            prompt += "# FORMAT EXACTLY LIKE THE EXAMPLE ABOVE\n\n"
            counter += 1

        prompt += "\n\n### Examples End\n"
        prompt += "\n\n## Input:\n"
        prompt += f"{text}\n"
        prompt += "## Input Response:\n<>\n"

        return prompt


    def balance_dataset_for_singleOutput(self, df, target_label):

        print(f'Number of labels before balancing: {df[target_label].sum()}')

        rus = RandomUnderSampler(random_state=42)

        target_data = df[target_label]
        X_data = df.drop(columns=[target_label])

        X_resampled, y_resampled = rus.fit_resample(X_data, target_data)

        dataframe = pd.DataFrame(X_resampled, columns=X_data.columns)
        dataframe[target_label] = y_resampled

        dataframe.reset_index(drop=True, inplace=True)

        print(f'Number of labels after balancing: {dataframe[target_label].sum()}')

        return dataframe

    def format_prompt_func(self, user_prompt, assistant_prompt, row, EOS_TOKEN):
        """Función optimizada para formatear un solo registro"""
        output = (
            f"<score>{row['[score]']}</score>"
        )

        message = [
            {
                "role": "user",
                "content": user_prompt.format(row['text'])
            },
            {
                "role": "assistant",
                "content": assistant_prompt.format(output, row['[COT]']) 
            }
        ]

        # print(f"Message: {message}")

        # return message
        template =  self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        return template;
        # return prompt.format(row['text'], row['[COT]'], output)

    def prepare_datasets(self, user_prompt, assistant_prompt, dataframe, EOS_TOKEN, test_size=None, random_state=42):
        """
        Divide el DataFrame y crea datasets para HuggingFace Trainer

        Args:
            dataframe: DataFrame de pandas con los datos
            test_size: Proporción para validation (default: 0.1)
            random_state: Semilla para reproducibilidad

        Returns:
            DatasetDict con splits train/validation incluyendo scores
        """
        # Dividir los datos
        if test_size != None:

            train_df, val_df = train_test_split(
                dataframe,
                test_size=test_size,
                random_state=random_state,
                stratify=dataframe['[score]'] if '[score]' in dataframe.columns else None
            )

            # Procesamiento eficiente con barra de progreso
            print("\nFormateando datos de entrenamiento...")


            train_data = []
            train_data_alpaca_json = []
            for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
                train_data.append({
                    "text": self.format_prompt_func(user_prompt, assistant_prompt, row, EOS_TOKEN),
                    "[score]": row['[score]'],
                })
                train_data_alpaca_json.append({
                    "instruction": user_prompt.format(row['text']),
                    "input": "",
                    "output": assistant_prompt.format(row['[COT]'], f"<score>{row['[score]']}</score>"),
                })

            print("\nFormateando datos de validación...")
            val_data = []
            val_data_alpaca_json = []
            for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
                val_data.append({
                    "text": self.format_prompt_func(user_prompt, assistant_prompt, row, EOS_TOKEN),
                    "[score]": row['[score]'],
                })
                val_data_alpaca_json.append({
                    "instruction": user_prompt.format(row['text']),
                    "input": "",
                    "output": assistant_prompt.format(row['[COT]'], f"<score>{row['[score]']}</score>"),
                })

            # Crear estructura de datasets
            dataset_dict = DatasetDict({
                'train': Dataset.from_list(train_data),
                'validation': Dataset.from_list(val_data)
            })
            
            #Guardar el json de entrenamiento y validación
            with open("train_data_alpaca.json", "w") as f:
                json.dump(train_data_alpaca_json, f, indent=4)
                f.close()
                
            with open("val_data_alpaca.json", "w") as f:
                json.dump(val_data_alpaca_json, f, indent=4)
                f.close()

        else:

            # Procesamiento eficiente con barra de progreso
            print("\nFormateando datos de entrenamiento...")
            train_data = []
            for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
                train_data.append({
                    "text": self.format_prompt_func(row),
                    "[score]": row['[score]'],
                })

            # Crear estructura de datasets
            dataset_dict = DatasetDict({
                'train': Dataset.from_list(train_data)
            })


        return dataset_dict

    def create_dataset(self, user_prompt, assistant_prompt, balance = True, balance_label = "[score]", test_size = 0.1, random_state = 42):

        try:

            if self.train_dataset is None:
                raise Exception("Train dataset is None")

            # Balance dataset
            if balance:
                self.trainer_test_dataset = self.balance_dataset_for_singleOutput(self.train_dataset, balance_label)
                print(f"{self.trainer_test_dataset[f'{balance_label}'].value_counts()}")
            else:
                self.trainer_test_dataset = self.train_dataset.copy()

            EOS_TOKEN = self.tokenizer.eos_token

            # Split dataset in train and validation
            self.trainer_test_dataset = self.prepare_datasets(
                user_prompt,
                assistant_prompt,
                self.trainer_test_dataset,
                EOS_TOKEN,
                test_size=test_size,
                random_state=random_state
            )

            print("***" * 50)
            print(f"Train dataset: {self.trainer_test_dataset['train'][0]}")
            #Count the number of examples in each split
            print(f"Train dataset size: {len(self.trainer_test_dataset['train'])}")
            # Count [score] values
            print(f"[score] values in train dataset: {Counter(self.trainer_test_dataset['train']['[score]'])}")
            if 'validation' in self.trainer_test_dataset:
                print(f"Validation dataset size: {len(self.trainer_test_dataset['validation'])}")
                print(f"[score] values in validation dataset: {Counter(self.trainer_test_dataset['validation']['[score]'])}")
            print("***" * 50)

        except Exception as e:
            print(e)
            print(traceback.format_exc())

    def estimate_max_seq_length(self, tokenizer, dataset, text_field="text", sample_size=None):
        lengths = []

        for i, example in tqdm(enumerate(dataset), total=len(dataset)):
            text = example[text_field]
            tokenized = tokenizer(text, truncation=False, return_tensors=None)
            lengths.append(len(tokenized["input_ids"]))

            if sample_size is not None and i >= sample_size:
                break

        max_len = max(lengths)
        p95_len = int(np.percentile(lengths, 95))
        return {
            "max_length": max_len,
            "p95_length": p95_len,
            "lengths": lengths
        }             # tell Optuna


    def load_or_create_study(self, sampler=None, previous_data=None):
        try:
            
            auto_sampler_module = optunahub.load_module(package="samplers/auto_sampler")
            
            with open(self.STUDY_PKL, "rb") as f:
                study = pickle.load(f)
            print("▶ Resumed existing study")
            
            # Print the sampler that was used
            print(f"Study is using sampler: {type(study.sampler).__name__}")
            
            # If it's an AutoSampler, print the chosen sampler
            if hasattr(study.sampler, '_sampler'):
                print(f"AutoSampler chose: {type(study.sampler._sampler).__name__}")
            
        except FileNotFoundError:
            
            if sampler == "TPESampler":
                sampler = TPESampler(seed=SEED)
            
            study = optuna.create_study(
                study_name=f"{wandb_project_name}_study",
                direction="maximize",
                storage=f"sqlite:///{wandb_project_name}_study.db",
                sampler=auto_sampler_module.AutoSampler(seed=SEED) if sampler is None else sampler,
                pruner=HyperbandPruner(max_resource=3, reduction_factor=3),
            )
            print("▶ New study created")
            
        
        if previous_data is not None:
            for trial_info in previous_data:
                f1_score = trial_info.pop("f1_score")
                
                frozen_trial = optuna.trial.create_trial(
                params=trial_info,
                values=[f1_score],
                distributions={
                    'r': IntDistribution(high=32, log=False, low=8, step=8), 
                    'lora_alpha': IntDistribution(high=64, log=False, low=8, step=8), 
                    'learning_rate': FloatDistribution(high=0.0005, log=True, low=1e-06, step=None), 
                    'num_train_epochs': IntDistribution(high=6, log=False, low=2, step=1), 
                    'per_device_train_batch_size': IntDistribution(high=8, log=False, low=1, step=1), 
                    'gradient_accumulation_steps': IntDistribution(high=8, log=False, low=1, step=1), 
                    'weight_decay': FloatDistribution(high=0.1, log=False, low=0.0, step=None), 
                    'lr_scheduler_type': CategoricalDistribution(choices=('linear', 'cosine', 'cosine_with_restarts')), 
                    'warmup_ratio': FloatDistribution(high=0.1, log=False, low=0.0, step=None)
                },
                # value=f1_score  # The objective value (f1_score)
                )
                
                study.add_trial(frozen_trial)
            
            print(f"▶ Added {len(previous_data)} previous trials to the study")
    
        return study

    def save_callback(self, study, trial):
        # Save the study
        with open(self.STUDY_PKL, "wb") as study_file:
            pickle.dump(study, study_file)
            study_file.close()
        
        # Save best config if this is the best trial
        if study.best_trial == trial:
            with open(f"{wandb_project_name}_best_config.json", "w") as config_file:
                json.dump(trial.params, config_file, indent=4)
                config_file.close()
                

    def objective(self, trial):
        
        
        # ─── search space ───
        r                           = trial.suggest_int("r", 8, 32, step=2)
        lora_alpha                  = trial.suggest_int("lora_alpha", 4, 64, step=4)
        learning_rate               = trial.suggest_float("learning_rate",  1e-6, 5e-4, log=True)
        num_epochs                  = trial.suggest_int("num_train_epochs",  2,6)
        per_device_train_batch_size = trial.suggest_int("per_device_train_batch_size", 1, 8)
        gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 8)
        weight_decay                = trial.suggest_float("weight_decay",  0.0, 0.1)
        lr_scheduler_type           = trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts"])
        warmup_ratio                = trial.suggest_float("warmup_ratio", 0.0, 0.1)
        
        target_module_options = [
            ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
            ("q_proj", "v_proj"),
            ("q_proj", "k_proj", "v_proj"),
            ("q_proj", "k_proj", "v_proj", "o_proj"),
            ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"),
            ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"),
        ]
        
        train_target_modules        = trial.suggest_categorical("target_module", target_module_options)
        lora_dropout = 0  # still fixed for unsloth models

        # ─── WandB per‑trial run ───
        wandb.init(
            project=wandb_project_name,
            group="optuna‑sweep",
            name=f"trial‑{trial.number}",
            reinit=True,
            config=trial.params,   # record the sampled hyper‑params
        )

        try:
            f1_score = self.train_lora_model(
                trial=trial,                # <── pass the Trial handle
                r=r, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                learning_rate=learning_rate,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                weight_decay=weight_decay,
                lr_scheduler_type=lr_scheduler_type,
                warmup_ratio=warmup_ratio,
                report_to="wandb",
                disable_tqdm=True,
                optuna_search=True,
                train_target_modules=train_target_modules
            )
            # wandb.log({"f1_score": f1_score})
            return f1_score                 #  Optuna maximises F1

        except optuna.TrialPruned:
            wandb.finish()                 # tidy up WandB
            raise                          # propagate pruning

        finally:
            wandb.finish()
            
    def init_search(self, trial = 200, initial_params = None, sampler=None, previous_data = None):
        
        """
        Initializes the Optuna study and starts the hyperparameter search.
        """
        
        # Login to WandB
        wandb.login(key=self.wandb_key)
        
        
        self.send_notification(f"Starting Optuna search")
        
        study = self.load_or_create_study(sampler=sampler, previous_data=previous_data) 
        
        if initial_params is not None:
            for params in initial_params:
                study.enqueue_trial(params)
                print(f"Enqueued trial with params: {params}")
        
        # Start the dashboard in a separate thread
        dashboard_thread = threading.Thread(
            target=run_server,
            kwargs={"storage": f"sqlite:///{wandb_project_name}_study.db", "port": 8080},
            daemon=True
        )
        dashboard_thread.start()
        
        study.optimize(
            self.objective,
            n_trials=trial,  # Number of trials to run
            callbacks=[self.save_callback],
        )
        
    def extract_trails(self, n_trials):
        """
        Extracts the trials from the study and saves them to a file.
        """
        study = self.load_or_create_study()
        
        # Get all trials
        trials = study.get_trials(deepcopy=False)
        
        # Get trials withi value > 0.69 
        filtered_trials = [trial for trial in trials if trial.values[0] > 0.56]
        
        for f_trial in filtered_trials:
            print(f'''$
                  "r":{f_trial.params['r']}, 
                  "lora_alpha":{f_trial.params['lora_alpha']},
                  "learning_rate":{f_trial.params['learning_rate']},
                  "num_train_epochs":{f_trial.params['num_train_epochs']},
                  "learning_rate":{f_trial.params['learning_rate']},
                  "per_device_train_batch_size":{f_trial.params['per_device_train_batch_size']},
                  "gradient_accumulation_steps":{f_trial.params['gradient_accumulation_steps']},
                  "weight_decay":{f_trial.params['weight_decay']},
                  "lr_scheduler_type":"{f_trial.params['lr_scheduler_type']}",
                  "warmup_ratio":{f_trial.params['warmup_ratio']},
                  "f1_score":{f_trial.values[0]},
                  #,
                  '''.replace("$", "{").replace("#", "}")
                  )
        
        
    class CoherenceEvalCallback(TrainerCallback):
        def __init__(self, tokenizer, eval_dataset, max_seq_length, num_samples=4, patience=3, optuna_search = True):
            self.tokenizer = tokenizer
            self.eval_dataset = eval_dataset
            self.num_samples = num_samples
            self.max_seq_length = max_seq_length
            self.score_pattern = re.compile(r"<score>(\d)</score>")
            self.patience = patience
            self.best_f1 = -np.inf
            self.patience_counter = 0
            self.stop_training = False
            self.metrics = {}
            self.generation_samples = []
            self.optuna_search = optuna_search

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            
            if self.stop_training:
                control.should_training_stop = True
                return control
            
            if metrics is None:
                metrics = {}
            
            model = kwargs['model']
            model.eval()
            
            FastLanguageModel.for_inference(model) # Enable native 2x faster inference

            with torch.no_grad():
                
                y_true = []
                y_pred = []
                
                print(f"Eval dataset lenght: {len(self.eval_dataset)}")
                #Coun the [score] values
                score_counts = Counter(self.eval_dataset['[score]'])
                print(f"[score] values: {score_counts}")
                
                class_with_min_data = min(score_counts, key=score_counts.get)
                min_data_count = score_counts[class_with_min_data]
                
                k = min(self.num_samples or min_data_count, min_data_count)
                
                print(f"Number of samples per class: {k}")
                
                # Shuffle and select k samples from each class
                class_1 = self.eval_dataset.filter(lambda x: x["[score]"] == 1).shuffle(seed=SEED).select(range(k))
                class_2 = self.eval_dataset.filter(lambda x: x["[score]"] == 2).shuffle(seed=SEED).select(range(k))
                class_3 = self.eval_dataset.filter(lambda x: x["[score]"] == 3).shuffle(seed=SEED).select(range(k))
                class_4 = self.eval_dataset.filter(lambda x: x["[score]"] == 4).shuffle(seed=SEED).select(range(k))
                class_5 = self.eval_dataset.filter(lambda x: x["[score]"] == 5).shuffle(seed=SEED).select(range(k))
                
                # Concatenate the datasets
                eval_subset = concatenate_datasets([class_1, class_2, class_3, class_4, class_5]).shuffle(seed=SEED)

                # print(f"Eval dataset lenght: {len(self.eval_dataset)}")
                
                for i in range(len(eval_subset)):
                    
                    text = eval_subset[i]["text"]
                    gt_score = eval_subset[i]["[score]"]
                    
                    y_true.append(gt_score)
                    
                    prompt = text.split("## Response:")[0].strip() + "\n## Response:"
                    pattern = r"^.*?<\|eot_id\|><\|start_header_id\|>user<\|end_header_id\|>\s*"
                    
                    prompt = re.sub(pattern, "", prompt, flags=re.DOTALL)
                    
                    # print(f"Prompt: {prompt}")
                    
                    # Generate response
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_seq_length,
                    ).to(model.device)

                    outputs = model.generate(
                        **inputs,
                        max_length=self.max_seq_length,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=False,
                        stopping_criteria=[ScoreStoppingCriteria(self.tokenizer)]
                    )
                    
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    print(f"Generated text: {generated_text}")

                    # Extract predicted score
                    try:
                        pred_score = int(self.score_pattern.search(generated_text.split("## Response:")[1]).group(1))
                        y_pred.append(pred_score)
                    except (AttributeError, ValueError):
                        # Si no se encuentra el patrón, se penaliza al modelo
                        y_pred.append(1)
                            
                    # print(f"GT Score: {gt_score}")
                    # print(f"Predicted Score: {y_pred[-1]}")

                    # Store 10 generated samples
                    if len(self.generation_samples) < 10:
                        self.generation_samples.append({
                            "prompt": prompt,
                            "generated_text": generated_text,
                            "gt_score": gt_score,
                            "pred_score": y_pred[-1]
                        })
                # print(y_pred)
                # print(y_true)
            # Calculate metrics
            current_f1_eval = f1_score(y_true, y_pred, average='macro')
            
            # Early stopping logic
            if current_f1_eval > self.best_f1:
                self.best_f1 = current_f1_eval
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    # print(f"\nEarly stopping triggered! F1 didn't improve for {self.patience} evaluations.")
                    self.stop_training = True
                    control.should_training_stop = True
            
            # Log metrics
            metrics.update({
                "eval_f1_score": current_f1_eval,
                "eval_patience_counter": self.patience_counter,
                "eval_best_f1": self.best_f1,
                "eval_accuracy": accuracy_score(y_true, y_pred),
                "eval_precision": precision_score(y_true, y_pred, average='macro'),
                "eval_recall": recall_score(y_true, y_pred, average='macro'),
                "eval_y_true": y_true,
                "eval_y_pred": y_pred,
            })
            
            self.metrics = metrics
            
            return control
            
            
    class CoherenceEvalCallbackOptuna(CoherenceEvalCallback):
        """
        Inherits every­thing you already wrote and adds:
        • trial.report()  →   pass metric to Optuna
        • trial.should_prune()  →   stop early & raise TrialPruned
        """
        def __init__(self, trial, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.trial = trial
            self.eval_step = 0          # keeps Hyperband’s “resource” counter

        def on_evaluate(self, args, state, control, metrics = None, **kwargs):
            super().on_evaluate(args, state, control, metrics, **kwargs)  # computes self.best_f1
            
            # If we are in the Optuna search, report the metrics
            if self.optuna_search:
                # ───────────────────────────── Optuna hooks ────────────────────────────
                self.trial.report(self.best_f1, step=self.eval_step)
                self.eval_step += 1

                # If Hyperband says stop → prune
                if self.trial.should_prune():
                    wandb.log({"trial/state": "pruned"})
                    control.should_training_stop = True      # tell Trainer
                    raise optuna.TrialPruned()  
                
            return control


    def train_lora_model(self,
                         trial,
                         r = 4, lora_alpha = 8,
                         lora_dropout = 0.5,
                         random_state = 42,
                         max_seq_length = 1024 * 5,
                         learning_rate = 2e-5,
                         per_device_train_batch_size = 16,
                         gradient_accumulation_steps = 1,
                         num_train_epochs = 1,
                         warmup_ratio = 0.1,
                         lr_scheduler_type = "cosine",
                         weight_decay = 0.1,
                         wandb_project_name = "fine-tuning-new-model",
                         early_stopping_patience = 3,
                         report_to = "none",
                         disable_tqdm = False,
                         optuna_search = False,
                         clean_start = True,
                         train_target_modules = None
                         ):
        
        """
        Train LoRA model with the given parameters.
        clean_start: If True, reload the model and tokenizer. If False, use the existing model and tokenizer.
        """

        try:

            # results = self.estimate_max_seq_length(self.tokenizer, self.trainer_test_dataset["train"], text_field="text")
            
            # max_seq_length = results["max_length"]
            # print(f"Max seq length: {max_seq_length}")

            # wandb.login(key=wandb_key)
            
            self.monitor_gpu_memory()
            
            if trial is not None and trial.number > 0: 
                # Clean up GPU memory explicitly
                if hasattr(self, 'model') and self.model is not None:
                    try:
                        del self.model
                    except:
                        pass
                    
                if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                    try:
                        del self.tokenizer
                    except:
                        pass
                    
                # Force garbage collection
                gc.collect()
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                print(f"Cleaned up memory for trial {trial.number}")
                time.sleep(60 * 2) # wait 120 seconds
                print(f"Waiting for the next trial...")
                
                self.monitor_gpu_memory()
            
            if clean_start:
                
                try:
                    # Delete existing model and tokenizer if they exist
                    if hasattr(self, 'model'):
                        del self.model
                    if hasattr(self, 'tokenizer'):
                        del self.tokenizer
                        
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    self.load_pretrained_model(max_seq_length=max_seq_length)
                except Exception as e:
                    print(f"Error in loading model: {e}")
                    print(traceback.format_exc())
                    return -np.inf

            try:
                
                # Check if the model already has adapters and remove them if necessary
                if hasattr(self.model, "peft_config"):
                    print("Model already has LoRA adapters. Removing them...")
                    # Convert back to base model
                    from peft import get_peft_model_state_dict
                    try:
                        # Get the state dict of the base model
                        base_model_state_dict = {}
                        for name, param in self.model.named_parameters():
                            if not param.requires_grad:
                                base_model_state_dict[name] = param
                        
                        # Create a fresh instance of the model
                        del self.model
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        # Reload the pretrained model
                        self.load_pretrained_model(max_seq_length=max_seq_length)
                    except Exception as e:
                        print(f"Error while removing LoRA adapters: {e}")
                        return -np.inf
                
                self.model = FastLanguageModel.get_peft_model(
                    model=self.model,
                    r = r,
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if train_target_modules is None else train_target_modules,
                    lora_alpha=lora_alpha,
                    bias="none",
                    lora_dropout=lora_dropout,
                    random_state = random_state,
                    use_rslora=False,
                    loftq_config=None,
                    finetune_vision_layers     = False, # Turn off for just text!
                    finetune_language_layers   = True,  # Should leave on!
                    finetune_attention_modules = True,  # Attention good for GRPO
                    finetune_mlp_modules       = True,  # SHould leave on always!
                )
            except Exception as e:
                print(f"-Error in loading LoRA model: {e}")
                return -np.inf
            
            self.monitor_gpu_memory()
                
            coherence_callback = self.CoherenceEvalCallbackOptuna(
                trial=trial,
                tokenizer=self.tokenizer,
                eval_dataset=self.trainer_test_dataset["validation"],
                max_seq_length=max_seq_length,
                num_samples=10,
                optuna_search=optuna_search,
            )
            
            train_callbacks = [
                # EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
                coherence_callback,
            ]

            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.trainer_test_dataset["train"],
                eval_dataset=self.trainer_test_dataset["validation"],
                max_seq_length=max_seq_length,
                dataset_num_proc=2,
                dataset_text_field="text",
                packing=False,
                args=TrainingArguments(
                    # remove_unused_columns=False,
                    per_device_train_batch_size=per_device_train_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    num_train_epochs=num_train_epochs,
                    warmup_ratio=warmup_ratio,
                    # max_steps=3,
                    learning_rate=learning_rate, #2e-4
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=10,
                    optim="adamw_8bit",
                    weight_decay=weight_decay,
                    lr_scheduler_type=lr_scheduler_type,
                    seed=SEED,
                    output_dir=wandb_project_name,
                    run_name=f"{wandb_project_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    report_to=report_to,
                    eval_strategy="epoch",
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_f1_score",
                    greater_is_better=True,
                    # per_device_eval_batch_size=16,
                    eval_accumulation_steps=1,
                    save_strategy="epoch",
                    save_total_limit=3,
                    disable_tqdm=disable_tqdm
                ),
                # dataset_kwargs={"input_columns": ["text", "[score]"]},
                callbacks=train_callbacks,
                )

            trainer_stats = unsloth_train(trainer)
            # trainer.train()
            
            if optuna_search and report_to == "wandb":
                
                self.send_notification(f"Optuna search completed")
                self.send_notification(f"eval_f1_score: {coherence_callback.metrics.get('eval_f1_score')}")
                self.send_notification(f"eval_accuracy: {coherence_callback.metrics.get('eval_accuracy')}")
                
                wandb.log({
                    "eval/f1_score": coherence_callback.metrics.get("eval_f1_score"),
                    "eval/accuracy": coherence_callback.metrics.get("eval_accuracy"),
                    "eval/precision": coherence_callback.metrics.get("eval_precision"),
                    "eval/recall": coherence_callback.metrics.get("eval_recall"),
                    "eval/patience_counter": coherence_callback.metrics.get("eval_patience_counter"),
                    "eval/best_f1": coherence_callback.metrics.get("eval_best_f1"),
                    "eval/y_true": wandb.Histogram(coherence_callback.metrics.get("eval_y_true")),
                    "eval/y_pred": wandb.Histogram(coherence_callback.metrics.get("eval_y_pred")),
                    "trial/state": "completed",
                    "eval/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=coherence_callback.metrics["eval_y_true"],
                        preds=coherence_callback.metrics["eval_y_pred"],
                    )
                })
                
                sample_table = wandb.Table(columns=["Prompt", "Generated", "Predicted", "True"])
                
                for sample in coherence_callback.generation_samples:
                    sample_table.add_data(
                        sample["prompt"],
                        sample["generated_text"],
                        sample["pred_score"],
                        sample["gt_score"]
                    )
                    
                wandb.log({"eval/generation_samples": sample_table})
                
            else:
                print(f"F1 Score: {coherence_callback.metrics.get('eval_f1_score'):.4f}")
                print(f"Accuracy: {coherence_callback.metrics.get('eval_accuracy'):.4f}")
                print(f"Precision: {coherence_callback.metrics.get('eval_precision'):.4f}")
                print(f"Recall: {coherence_callback.metrics.get('eval_recall'):.4f}")
                print(f"Patience Counter: {coherence_callback.metrics.get('eval_patience_counter')}/{early_stopping_patience}")
                print(f"Best F1 Score: {coherence_callback.metrics.get('eval_best_f1'):.4f}")
            
            if optuna_search:
                return coherence_callback.metrics.get('eval_f1_score')
        
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e):
                print(f"CUDA out of memory error in trial {trial.number if trial else 'N/A'}: {e}")
                
                # Clean up
                if hasattr(self, 'model'):
                    del self.model
                if hasattr(self, 'tokenizer'):
                    del self.tokenizer
                    
                gc.collect()
                torch.cuda.empty_cache()
                
                self.send_notification(f"CUDA out of memory error in trial {trial.number if trial else 'N/A'}: {e}")
                
                # Return a value that indicates failure to Optuna
                return -np.inf
            else:
                print(f"RuntimeError: {e}")
                print(traceback.format_exc())
                return -np.inf
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.send_notification(f"Error in trial {trial.number if trial else 'N/A'}: {e}")
            return -np.inf
        finally:
            gc.collect()
            torch.cuda.empty_cache()


    def evaluate_lora_mode(
        self, 
        prompt, 
        custom_dataframe=None, 
        batch_size=8, 
        max_new_tokens=None, 
        temperature=1.0, 
        top_k=50, 
        top_p=1.0, 
        do_sample=False, 
        streaming=False, 
        stop_on_first_input=False
    ):
        """
        Evaluates the model on the test dataset using the provided prompt.
        """
        try:
            # Preparar prompts desde custom_dataframe o test_dataset
            if custom_dataframe is not None:
                prompts = [prompt.format(row['text']) for _, row in custom_dataframe.iterrows()]
                labels = [row['[score]'] for _, row in custom_dataframe.iterrows()]
                original_texts = [row['text'] for _, row in custom_dataframe.iterrows()]
                stories_ids = [row['[ID]'] for _, row in custom_dataframe.iterrows()]
            else:
                prompts = [prompt.format(row['text']) for _, row in self.test_dataset.iterrows()]
                labels = [row['[score]'] for _, row in self.test_dataset.iterrows()]
                original_texts = [row['text'] for _, row in self.test_dataset.iterrows()]
                stories_ids = [row['[ID]'] for _, row in self.test_dataset.iterrows()]

            # Construir mensajes para chat-template
            messages_list = [[{"role": "user", "content": p}] for p in prompts]

            # Tokenizar con apply_chat_template y reasoning_effort bajo
            encodings = [
                self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    reasoning_effort="low"
                ).to(self.model.device)
                for messages in messages_list
            ]

            # Empaquetar como dataset
            dataset = []
            for prompt, label, text, story_id in zip(prompts, labels, original_texts, stories_ids):
                messages = [{"role": "user", "content": prompt}]
                encoding = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    reasoning_effort="low"
                ).to(self.model.device)

                dataset.append((
                    encoding["input_ids"].squeeze(0),   # quitar dimensión batch
                    encoding["attention_mask"].squeeze(0),
                    label,
                    text,
                    story_id
                ))

            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: list(zip(*x)))
            
            print(f"Evaluating on {len(dataset)} samples...")

            y_true, y_pred, tp, fp, tn, fn = [], [], [], [], [], []

            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids_batch, attention_mask_batch, labels_batch, original_texts_batch, stories_ids_batch = batch

                input_ids_batch = torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                attention_mask_batch = torch.nn.utils.rnn.pad_sequence(attention_mask_batch, batch_first=True, padding_value=0)


                inputs = {
                    "input_ids": input_ids_batch.to(self.model.device),
                    "attention_mask": attention_mask_batch.to(self.model.device),
                }

                streamer = None
                if streaming:
                    streamer = TextStreamer(self.tokenizer, skip_prompt=True)

                generate_kwargs = {
                    **inputs,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "do_sample": do_sample,
                    "max_length": max_new_tokens,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "stopping_criteria": StoppingCriteriaList([ScoreStoppingCriteria(self.tokenizer)])
                }
                if streamer is not None:
                    generate_kwargs["streamer"] = streamer

                outputs = self.model.generate(**generate_kwargs)
                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for text, true_score, original_text, story_id in zip(decoded, labels_batch, original_texts, stories_ids):
                    print("*" * 50)
                    print(f"Text: {text.split('Response:')[1] if 'Response:' in text else text}")
                    print("*" * 50)

                    pred_score = self.get_output_score(text.rsplit("Response:", 10)[-1])
                    if pred_score is None:
                        print("Score is None, fallback to shifting true score")
                        pred_score = (true_score + 1) if true_score < 5 else 1

                    print(f"Y_true: {true_score}, Y_pred: {pred_score}")

                    # Métricas
                    if true_score == 1 and pred_score == 1:
                        tp.append(story_id)
                    elif true_score == 0 and pred_score == 1:
                        fp.append(story_id)
                    elif true_score == 0 and pred_score == 0:
                        tn.append(story_id)
                    elif true_score == 1 and pred_score == 0:
                        fn.append(story_id)

                    y_true.append(true_score)
                    y_pred.append(pred_score)

                    gen_text = text.rsplit("Response:", 10)[-1]
                    result = self.highlight_problematic_sentences(original_text, gen_text, score=pred_score)
                    display(result)

                if stop_on_first_input:
                    break

            print(f"Y_true: {y_true}")
            print(f"Y_pred: {y_pred}")
            print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

            numeric_y_true = [int(y) for y in y_true]
            return numeric_y_true, y_pred

        except Exception as e:
            print(e)
            print(traceback.format_exc())

            
    def show_metrics(self, y_true, y_pred):
        
        """ 
        Funtion to plot the confusion matrix and classification report.
        """
        
        if type(y_true[0]) == torch.Tensor:
            y_true = [tensor.item() for tensor in y_true]
        
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix using seaborn
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión sobre el conjunto de prueba')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        plt.show()

        print(classification_report(y_true, y_pred))
        
        
    def save_lora_model(self, model_name, save_path):
        """
        Save the LoRA model to the specified path.
        """
        if hasattr(self.model, "peft_config"):
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"LoRA model saved to {save_path}")
        else:
            print("Model is not a LoRA model. Cannot save.")
            
            
    def load_lora_model(self, load_path, max_seq_length=1024 * 5, dtype=None, load_in_4bit=True):
        
        """
        Load the LoRA model from the specified path.
        """
        try:
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=load_path,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=bool(load_in_4bit)
            )
            
            FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference
            
            print(f"LoRA model loaded from {load_path}")
            
        except Exception as e:
            print(f"Error loading LoRA model: {e}")
            print(traceback.format_exc())
            
    def generate_response(self, prompt, max_new_tokens=100, temperature=1.0, top_k=50, top_p=1.0, do_sample=False, streaming=True):
        
        # Tokenize inputs in batches
        encodings = self.tokenizer(prompt, truncation=True, padding=True, return_tensors="pt")
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        
        inputs = {
            "input_ids": input_ids.to(self.model.device),
            "attention_mask": attention_mask.to(self.model.device),
        }
        
        streamer = None
        if streaming:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            
        generate_kwargs = {
            **inputs,
            # "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": do_sample,
            "max_length": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        if streamer is not None:
            generate_kwargs["streamer"] = streamer
        
        outputs = self.model.generate(**generate_kwargs)
        
        if not streaming:
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print(f"Decoded: {decoded}")
    
    def highlight_problematic_sentences(self, texto_original, texto_generado, score=None):
        """
        Colorea las oraciones problemáticas del texto original basándose en el análisis del LLM.
        
        Args:
            texto_original (str): El texto original a analizar
            texto_generado (str): El análisis generado por el LLM con las oraciones problemáticas
            
        Returns:
            HTML: Objeto HTML para mostrar en Jupyter/Python con las oraciones coloreadas
        """
        
        # Definir colores para cada tipo de problema
        color_map = {
            'r1': "#cce1ff",  # Rojo claro - No tiene sentido
            'r2': '#ffffcc',  # Amarillo claro - Entidad no introducida
            'r3': '#ccffcc',  # Verde claro - Relación sin sentido
            'r4': '#ccccff',  # Azul claro - Inconsistente con datos previos
            'r5': '#ffccff',  # Magenta claro - Inconsistente con conocimiento mundial
            'r6': '#ffcc99',  # Naranja claro - No relevante al título
            'r7': '#ccffff'   # Cian claro - No relevante a datos anteriores
        }
        
        # Extraer oraciones problemáticas del texto generado
        problematic_sentences = []
        
        # Buscar la sección de "Problematic sentences"
        problematic_section_match = re.search(
            r'2\.\s*Problematic sentences:.*?(?=3\.|$)', 
            texto_generado, 
            re.DOTALL | re.IGNORECASE
        )
        
        if not problematic_section_match:
            print("No problematic section found")
        else:
            problematic_section = problematic_section_match.group(0)
            print(f"Problematic section found:\n{problematic_section[:200]}...\n")
            
            # Intentar formato original primero: "oración" seguido de - Reason: razón - Tags: <r1>1</r1>
            sentence_pattern_original = r'-\s*"([^"]+)"[^-]*?-\s*Reason:\s*([^-]+?)-\s*Tags:\s*<(r\d)>\d</r\d>'
            matches_original = re.findall(sentence_pattern_original, problematic_section, re.DOTALL)
            
            if matches_original:
                print("=== USING ORIGINAL FORMAT ===")
                for i, match in enumerate(matches_original):
                    sentence = match[0].strip()
                    reason = match[1].strip()
                    tag = match[2]
                    print(f"Sentence {i+1}: '{sentence}'")
                    print(f"Reason {i+1}: '{reason}'")
                    print(f"Tag {i+1}: '{tag}'")
                    print("-" * 50)
                    problematic_sentences.append((sentence, tag))
            else:
                # Intentar nuevo formato: -The sentence 'oración' descripción <r1>1</r1> más descripción <r5>1</r5>
                print("=== USING NEW FORMAT ===")
                
                # Buscar todas las líneas que empiecen con -The sentence
                sentence_lines = re.findall(
                    r'-The sentence \'([^\']+)\'([^-]*?)(?=-The|$)',
                    problematic_section,
                    re.DOTALL
                )
                
                print(f"Found {len(sentence_lines)} sentence lines")
                
                for i, sentence_line in enumerate(sentence_lines):
                    sentence = sentence_line[0].strip()
                    description_with_tags = sentence_line[1].strip()
                    
                    print(f"Sentence {i+1}: '{sentence}'")
                    print(f"Full description: '{description_with_tags[:100]}...'")
                    
                    # Extraer todos los tags de la descripción
                    tag_matches = re.findall(r'<(r\d)>\d</r\d>', description_with_tags)
                    
                    print(f"Tags found: {tag_matches}")
                    
                    if tag_matches:
                        # Solo usar el último tag detectado como solicitas
                        last_tag = tag_matches[-1]
                        print(f"Using last tag: '{last_tag}'")
                        problematic_sentences.append((sentence, last_tag))
                    else:
                        print("No tags found for this sentence")
                    
                    print("-" * 50)
                
                # Si no encontró con el patrón anterior, intentar otro patrón más flexible
                if not sentence_lines:
                    print("Trying alternative pattern...")
                    alternative_lines = re.findall(
                        r'-The sentence \'([^\']+)\'[^<]*?(<r\d>\d</r\d>(?:[^<-]*?<r\d>\d</r\d>)*)',
                        problematic_section,
                        re.DOTALL
                    )
                    
                    print(f"Alternative pattern found {len(alternative_lines)} sentences")
                    
                    for i, alt_match in enumerate(alternative_lines):
                        sentence = alt_match[0].strip()
                        tags_part = alt_match[1]
                        
                        print(f"Alt Sentence {i+1}: '{sentence}'")
                        print(f"Tags part: '{tags_part}'")
                        
                        # Extraer todos los tags
                        tag_matches = re.findall(r'<(r\d)>\d</r\d>', tags_part)
                        
                        print(f"Tags found: {tag_matches}")
                        
                        if tag_matches:
                            # Solo usar el último tag detectado
                            last_tag = tag_matches[-1]
                            print(f"Using last tag: '{last_tag}'")
                            problematic_sentences.append((sentence, last_tag))
                        
                        print("-" * 50)
        
        # Si no se encontraron oraciones problemáticas, retornar texto original
        if not problematic_sentences:
            return HTML(f'<div style="font-family: Arial, sans-serif; line-height: 1.6; padding: 20px;">{texto_original}</div>')
        
        print(f"Found {len(problematic_sentences)} problematic sentences")
        
        # Crear HTML con oraciones coloreadas
        html_texto = texto_original
        
        # Ordenar por longitud descendente para evitar reemplazos parciales
        problematic_sentences.sort(key=lambda x: len(x[0]), reverse=True)
        
        for sentence, tag in problematic_sentences:
            color = color_map.get(tag, '#f0f0f0')  # Color por defecto si no se encuentra el tag
            
            # Escapar caracteres especiales para regex
            escaped_sentence = re.escape(sentence)
            
            # Crear el reemplazo con color y tooltip
            tooltip_text = self.get_reason_description(tag)
            replacement = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; border-left: 4px solid {self.get_border_color(tag)};" title="{tooltip_text}">{sentence}</span>'
            
            # Reemplazar la oración en el texto
            html_texto = re.sub(escaped_sentence, replacement, html_texto, flags=re.IGNORECASE)
        
        # Crear leyenda
        legend_html = self.create_legend(problematic_sentences, color_map)
        
        resultado = f"Clase predicha : {score}" if score is not None else "Clase predicha: No disponible"
        
        # Combinar texto coloreado con leyenda
        final_html = f"""
        <div style="font-family: Arial, sans-serif; line-height: 1.8; padding: 20px; max-width: 800px;">
            <h3 style="color: #333; margin-bottom: 20px;">Texto con Oraciones Problemáticas Resaltadas</h3>
            <h4>{resultado}</h4>
            <div style="background-color: #f9f9f9; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                {html_texto}
            </div>
            {legend_html}
        </div>
        """
        
        return HTML(final_html)

    def get_reason_description(self, tag):
        """Retorna la descripción del tipo de problema"""
        descriptions = {
            'r1': 'La oración no tiene sentido',
            'r2': 'La oración discute una entidad que no ha sido introducida aún',
            'r3': 'La relación entre esta oración y las anteriores no tiene sentido',
            'r4': 'La oración contiene información inconsistente con datos presentados previamente',
            'r5': 'La oración contiene información inconsistente con el conocimiento del mundo',
            'r6': 'La oración no es relevante para el título',
            'r7': 'La oración no es relevante para los datos anteriores en la historia'
        }
        return descriptions.get(tag, 'Problema no identificado')

    def get_border_color(self, tag):
        """Retorna un color más oscuro para el borde"""
        border_colors = {
            'r1': "#7eb2fc",  # Rojo
            'r2': '#ffff99',  # Amarillo
            'r3': '#99ff99',  # Verde
            'r4': '#9999ff',  # Azul
            'r5': '#ff99ff',  # Magenta
            'r6': '#ff9966',  # Naranja
            'r7': '#99ffff'   # Cian
        }
        return border_colors.get(tag, '#cccccc')

    def create_legend(self, problematic_sentences, color_map):
        """Crea una leyenda con los tipos de problemas encontrados"""
        found_tags = list(set([tag for _, tag in problematic_sentences]))
        
        if not found_tags:
            return ""
        
        legend_items = []
        for tag in sorted(found_tags):
            color = color_map.get(tag, '#f0f0f0')
            border_color = self.get_border_color(tag)
            description = self.get_reason_description(tag)
            
            legend_item = f"""
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="width: 20px; height: 20px; background-color: {color}; border-left: 4px solid {border_color}; border-radius: 3px; margin-right: 10px;"></div>
                <span style="font-size: 14px;"><strong>{tag.upper()}:</strong> {description}</span>
            </div>
            """
            legend_items.append(legend_item)
        
        legend_html = f"""
        <div style="margin-top: 20px; padding: 15px; background-color: #ffffff; border: 1px solid #ddd; border-radius: 8px;">
            <h4 style="color: #333; margin-bottom: 15px; margin-top: 0;">Razones de incoherencia detectadas:</h4>
            {''.join(legend_items)}
        </div>
        """
        
        return legend_html
    
    def monitor_gpu_memory(self):
        """Print GPU memory usage for monitoring."""
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                free = total_memory - allocated
                
                print(f"GPU {i}: Total: {total_memory:.2f}GB | "
                    f"Reserved: {reserved:.2f}GB | "
                    f"Allocated: {allocated:.2f}GB | "
                    f"Free: {free:.2f}GB")
        else:
            print("No CUDA devices available")
    
