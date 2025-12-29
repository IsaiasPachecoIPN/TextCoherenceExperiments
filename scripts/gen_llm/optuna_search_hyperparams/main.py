import sys 

module_dir = "../utils"
sys.path.append(os.path.abspath(module_dir))

try:
    from llm_utils import *
except ImportError:
    print(f"Error: Could not import 'utils'. Make sure utils.py is in '{module_dir}'")
    sys.exit(1)

# Choose the model to use

# model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
# model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit"
# model_name = "unsloth/phi-4-unsloth-bnb-4bit"
# model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
# model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
# model_name= "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
# model_name = "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit"
# model_name = "unsloth/gemma-3-12b-it-unsloth-bnb-4bit"
# model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
model_name = "unsloth/SmolLM2-1.7B-Instruct-bnb-4bit"

TEST_DATASET_PATH = "./testDataset.csv"
TRAIN_DATASET_PATH = "./trainDataset.csv"

llm_model = LLMModel(model_name=model_name)

llm_model.load_pretrained_model(max_seq_length=1024 * 5)

llm_model.load_test_csv_dataset(TEST_DATASET_PATH)

llm_model.load_train_csv_dataset(TRAIN_DATASET_PATH)

user_prompt = """
You are an AI system tasked with evaluating the textual coherence of a short story based on its title and content. Your job is to perform a sentence-by-sentence analysis of the story and identify any coherence issues. Each problematic sentence must be tagged with the appropriate reason code, explained below.

## Instruction:
Analyze the given story sentence by sentence. Identify any coherence issues and assign the relevant tags from the list below. Each problematic sentence should include a short explanation and be marked with one or more of the following reason tags:

### Reason Tags:
- <r1>1</r1> - the sentence doesn't make sense
- <r2>1</r2> - the sentence discusses an entity which has not been introduced yet
- <r3>1</r3> - the relation between this sentence and previous ones doesn't make sense
- <r4>1</r4> - the sentence contains information inconsistent with previous presented data
- <r5>1</r5> - the sentence contains information inconsistent with the knowledge about the world
- <r6>1</r6> - the sentence is not relevant to the title
- <r7>1</r7> - the sentence is not relevant to previous data in the story

IMPORTANT: You MUST follow the EXACT format shown below. Analyze each sentence individually and assign appropriate reason tags (<r1> through <r7>) to each problematic sentence. Use the tags exactly as shown.

Finally, you will provide the score of the text coherence using the following format.
## Scoring criteria:
- <score>1</score> if the story is not coherent at all.
- <score>2</score> if the story is not coherent.
- <score>3</score> if the story is maybe yes, maybe not coherent.
- <score>4</score> if the story is coherent.
- <score>5</score> if the story is definitely coherent.

## Response Format:
1. Overall Coherence:
   - [Brief assessment of whether the story is coherent or not]
   
2. Problematic sentences:
   - "[Exact problematic sentence]" !Important: don not repeat the sentences, each sentence must be unique!
     - Reason: [Brief explanation]
     - Tags: <rX>1</rX>
     
3. Conclusion:
   - [Summary conclusion about the story’s coherence]
   
4. Score:
   - <score>[1,2,3,4 or 5]</score>

## Input:
{}
"""

assistant_prompt = """
## Response:
{}
4. Score:
{}"""

# Split the dataset into train and test sets
llm_model.create_dataset(user_prompt, assistant_prompt, test_size=0.2, balance=False)


# print("Initializing the search...")
llm_model.init_search(trial=100, initial_params=None)

# llm_model.extract_trails(0)