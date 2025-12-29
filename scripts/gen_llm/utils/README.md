# LLM Fine-Tuning and Optimization Utility

This library provides a high-level wrapper around **Unsloth**, **Hugging Face TRL**, and **Optuna** to streamline the fine-tuning, hyperparameter optimization, and evaluation of Large Language Models (LLMs). It is designed specifically for classification and coherence scoring tasks, incorporating automated hyperparameter search, LoRA adapter management, and custom evaluation callbacks.

## Key Features

* **Efficient Fine-Tuning:** Leveraging Unsloth for optimized 4-bit quantization and faster backpropagation.
* **Hyperparameter Optimization (HPO):** Integrated Optuna study management with SQLite storage and WandB tracking to optimize LoRA rank, alpha, learning rates, and schedulers.
* **Custom Evaluation Callbacks:** Real-time calculation of F1-score, Accuracy, Precision, and Recall during training steps to drive early stopping and pruning.
* **Few-Shot Experimentation:** Native support for N-shot inference to benchmark zero-shot vs. few-shot performance before or after fine-tuning.
* **Coherence Analysis:** Visualization tools to highlight problematic sentences in generated outputs based on specific error tags (e.g., consistency, relevance).
* **Experiment Tracking:** Full integration with Weights & Biases (WandB) for logging metrics, confusion matrices, and generation samples.

## Installation

Ensure you have a Python environment compatible with PyTorch (CUDA support recommended). The following core dependencies are required:

```bash
pip install torch unsloth transformers trl peft optuna optuna-dashboard wandb scikit-learn pandas matplotlib seaborn

```

*Note: Access to restricted models (e.g., Llama, Mistral) requires a Hugging Face token.*

## Dataset Structure

The library expects input data in CSV format. The internal data processor is designed to handle supervised fine-tuning (SFT) for tasks involving Chain-of-Thought (CoT) reasoning and scoring.

### Required Columns

The input CSV files must contain the following columns:

| Column Name | Data Type | Description |
| --- | --- | --- |
| `text` | String | The input text or query to be processed by the model. |
| `[score]` | Integer | The target classification label or score (e.g., 1-5). |
| `[COT]` | String | The Chain-of-Thought reasoning explaining the score. |
| `[ID]` | String/Int | (Optional) Unique identifier for the data point. |

### Internal Formatting

The library converts this data into a chat template format (User/Assistant) automatically:

* **User:** Contains the instruction and the input `text`.
* **Assistant:** Contains the `[COT]` reasoning followed by the final `<score>X</score>` tag.

## Usage Guide

### 1. Initialization and Model Loading

Initialize the wrapper with the desired model path. The library supports 4-bit loading by default for memory efficiency.

```python
from llm_utils import LLMModel

# Initialize the class
llm = LLMModel(model_name="unsloth/SmolLM2-1.7B-Instruct")

# Load the pretrained model and tokenizer
llm.load_pretrained_model(
    max_seq_length=2048, 
    load_in_4bit=True
)

```

### 2. Data Preparation

Load your training and testing datasets. The `create_dataset` method handles class balancing (undersampling), train/validation splitting, and prompt formatting.

```python
# Load raw CSVs
llm.load_train_csv_dataset("path/to/train_data.csv")
llm.load_test_csv_dataset("path/to/test_data.csv")

# Define prompt templates
user_prompt = "Analyze the coherence of the following text: {}"
assistant_prompt = "Analysis: {} \nScore: {}"

# Process datasets
llm.create_dataset(
    user_prompt=user_prompt,
    assistant_prompt=assistant_prompt,
    balance=True,          # Apply RandomUnderSampler to balance classes
    balance_label="[score]",
    test_size=0.1          # 10% for validation
)

```

### 3. Hyperparameter Optimization (Optuna)

The library can execute an automated search for the optimal hyperparameters. This process utilizes the `TPESampler` or `AutoSampler` and prunes unpromising trials using `HyperbandPruner`.

```python
# Initialize Optuna search
llm.init_search(
    trial=50,              # Number of trials to run
    sampler=None,          # Default is TPESampler
    initial_params=None    # Optional list of dicts with starting parameters
)

```

**Search Space:**
The `objective` function automatically optimizes:

* **LoRA Config:** `r` (8-32), `lora_alpha` (4-64), `target_modules`.
* **Training Config:** `learning_rate`, `num_train_epochs`, `batch_size`, `gradient_accumulation_steps`.
* **Optimizer:** `weight_decay`, `lr_scheduler_type` (linear, cosine, restarts), `warmup_ratio`.

### 4. Fine-Tuning

Once the best parameters are identified (or using manual defaults), initiate the training process.

```python
llm.train_lora_model(
    r=16,
    lora_alpha=32,
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    clean_start=True,      # Reloads base model to ensure clean state
    report_to="wandb",     # Log metrics to Weights & Biases
    wandb_project_name="Production-FineTune-v1"
)

```

### 5. Evaluation

Evaluate the fine-tuned model on the test dataset. The evaluation generates a confusion matrix and highlights problematic output sentences based on specific error tags (e.g., `<r1>`, `<r2>`).

```python
y_true, y_pred = llm.evaluate_lora_mode(
    prompt="## Input:\n{}\n## Response:",
    batch_size=8,
    max_new_tokens=512,
    temperature=0.1
)

# Visualize metrics
llm.show_metrics(y_true, y_pred)

```

### 6. N-Shot Experimentation

Run baseline benchmarks using N-shot prompting without training to establish a performance baseline.

```python
llm.n_shot_experiment(
    promt="Classify this text...",
    n_shot=3,              # Number of examples per class to include in context
    max_new_tokens=256,
    verbose=True
)

```

## Advanced Configuration

### CoherenceEvalCallback

The training loop includes a custom callback `CoherenceEvalCallback`. Unlike standard loss evaluation, this callback:

1. Pauses training at the end of every epoch.
2. Generates full text responses for a subset of the validation data.
3. Parses the generated `<score>` tags.
4. Calculates `F1-score`, `Accuracy`, `Precision`, and `Recall`.
5. Triggers Early Stopping if the generation metrics do not improve.

### Memory Management

The library includes explicit garbage collection and CUDA cache clearing mechanisms (`monitor_gpu_memory`, `gc.collect()`) to handle the memory constraints typical of iterative hyperparameter search on consumer or cloud GPUs.

## Output Artifacts

* **WandB:** Logs training curves, confusion matrices, and tables containing sample model generations versus ground truth.
* **SQLite Database:** Stores the state of the Optuna study (`{project_name}_study.db`).
* **Models:** Saves the best LoRA adapters to the specified output directory.
* **Analysis:** HTML visualizations of problematic sentences are generated during evaluation for qualitative analysis. 