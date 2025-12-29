
# LLM Hyperparameter Optimization (Optuna Search)

This script automates the hyperparameter tuning process for Large Language Model (LLM) fine-tuning using **Optuna** and **Unsloth**. It performs a Bayesian search to identify the optimal configuration for LoRA adapters, learning rates, and training schedules, optimizing specifically for text coherence and classification tasks.

## Prerequisites

### 1. File Structure

Ensure your directory is structured as follows to allow the script to import the utility module and locate datasets:

```
project_root/
├── utils/
│   └── llm_utils.py         # The core library
├── scripts/
│   └── optuna_search.py     # This script
├── trainDataset.csv         # Training data
└── testDataset.csv          # Testing/Validation data

```

### 2. Environment Variables

The underlying `llm_utils` library requires a `.env` file or exported environment variables for tracking:

* `WANDB_API_KEY`: Your Weights & Biases API key.
* `WANDB_PROJECT`: The name of your WandB project.
* `TELEGRAM_BOT_TOKEN`: Your Telegram bot token.
* `TELEGRAM_CHAT_ID`: Your Telegram chat id.

## Configuration

Before running the script, you can modify the following variables directly in the python file:

* **Model Selection:** Uncomment the desired `model_name` variable to switch between architectures (e.g., Llama 3, Mistral, Qwen, Gemma).
* **Dataset Paths:** Update `TRAIN_DATASET_PATH` and `TEST_DATASET_PATH` if your CSV files are located elsewhere.
* **Search Budget:** Adjust `trial=100` in `llm_model.init_search()` to increase or decrease the number of hyperparameter combinations to test.

## Usage

Run the script from your terminal:

```bash
python optuna_search.py

```

The script will:

1. Load the specified 4-bit quantized model.
2. Load and process the datasets into the required Chat Template format.
3. Initialize the Optuna study.
4. Begin the iterative training and evaluation process.

## Monitoring the Search

You can monitor the progress and performance of the hyperparameter search in real-time using two primary tools:

### 1. Optuna Dashboard (Real-Time Visualization)

The script automatically initializes a local Optuna Dashboard server. This provides an interactive UI to visualize parameter relationships and trial history.

* **Access:** Open your browser and navigate to `http://127.0.0.1:8080`
* **Features:**
* **Graph:** Visualize the optimization history.
* **Parallel Coordinate Plot:** See relationships between high-dimensional hyperparameters (e.g., how `learning_rate` and `lora_alpha` affect the score).
* **Importance:** View which hyperparameters have the most significant impact on model performance.



### 2. Weights & Biases (WandB)

Deep learning metrics are logged to the cloud.

* **Access:** Go to your project dashboard at [wandb.ai](https://www.google.com/search?q=https://wandb.ai/).
* **Features:**
* **Loss Curves:** Track training and validation loss for every trial.
* **Evaluation Metrics:** Monitor the evolution of `F1-Score`, `Accuracy`, and `Coherence` scores.
* **Generated Samples:** View actual text outputs from the model during validation steps to qualitatively assess performance.



## Retrieving Results

### Best Parameters

Once the search is complete (or if you stop it early), the optimal hyperparameters are stored in two locations:

1. **Console Output:** The script will print the best trial's parameters and objective value (F1 Score) to the terminal.
2. **Database:** The study state is saved to a SQLite database (`*.db` file) in the working directory. You can resume the study later or query it using standard Optuna CLI tools.
3. **JSON Artifact:** A JSON file named `{project_name}_best_config.json` is automatically generated, containing the winning configuration.

### Example Output

```json
{
    "r": 16,
    "lora_alpha": 32,
    "learning_rate": 0.0002,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2
}

```