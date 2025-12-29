import torch
import pandas as pd
from unsloth import FastLanguageModel
from transformers import TextStreamer
import random
import re
import os
import datetime

# --- CONFIGURATION ---
# Set the number of stories you want to generate here
N_STORIES = 140

# Name of the base model to use (Instruct version recommended for following prompts)
BASE_MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit" 
# BASE_MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit" # Alternative

# Output filename
OUTPUT_CSV = "generated_stories.csv"
# ---------------------

def load_base_model():
    """Loads the base model using Unsloth without LoRA adapters."""
    max_seq_length = 2048 
    dtype = None          
    load_in_4bit = True   # Use 4bit quantization to reduce memory usage

    print(f"⏳ Loading base model: {BASE_MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL_NAME,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    
    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model)
    print("✅ Model loaded successfully (Base Model Only).")
    return model, tokenizer

def generate_stories(model, tokenizer, n=5, max_new_tokens=1024, output_file="stories.csv"):
    """Generates n stories and saves them to a CSV file."""
    
    print(f"🚀 Starting generation of {n} stories...")
    
    stories = []
    existing_titles = set()
    existing_content_hashes = set()
    
    
    themes = [
        "adventure in a fantasy world", "science fiction exploration", 
        "historical fiction", "mystery thriller", "romantic comedy",
        "supernatural horror", "philosophical dilemma", "coming of age",
        "dystopian future", "magical realism", "heist story", "detective noir",
        "epic fantasy", "cyberpunk", "space opera", "alternate history",
        "psychological thriller", "urban fantasy", "fairy tale retelling",
        "superhero origin", "post-apocalyptic survival", "western frontier",
        "time travel paradox", "political intrigue", "murder mystery",
        "family drama", "supernatural comedy", "steampunk adventure",
        "nautical adventure", "spy thriller", "gothic horror", "magical academy",
        "folklore inspired", "ancient mythology", "courtroom drama", 
        "biographical fiction", "portal fantasy", "animal perspective",
        "artificial intelligence", "religious allegory", "sports drama"
    ]
    
    structures = [
        "linear narrative", "non-linear with flashbacks", 
        "multiple perspectives", "frame story", "epistolary format",
        "unreliable narrator", "stream of consciousness", "vignettes",
        "circular narrative", "hero's journey"
    ]
    
    attempts = 0
    max_attempts = n * 4  # Safety limit to prevent infinite loops
    
    while len(stories) < n and attempts < max_attempts:
        attempts += 1
        
        # 1. Prepare Prompt
        theme = random.choice(themes)
        structure = random.choice(structures)
        rand_max_new_tokens = random.randrange(512, max_new_tokens)
        
        prompt_text = (
            f"Generate a compelling and creative story with the theme of '{theme}' "
            f"using a '{structure}' structure. "
        )
        
        # Add random constraints for diversity
        if random.random() < 0.5: prompt_text += "Include a surprising twist. "
        if random.random() < 0.5: prompt_text += "Set in an unusual location. "
        if random.random() < 0.3: prompt_text += "Incorporate elements of humor. "
            
        prompt_text += "\n\nUse the format:\nTitle: [Generated Title]\nText: [Generated Story]\n\nEnd the story with a period."
        
        messages = [{"role": "user", "content": prompt_text}]
        
        input_ids = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to("cuda")
        
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        
        # 2. Generate
        # Using TextStreamer to print progress to console (optional, can be removed to be silent)
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=rand_max_new_tokens,
            temperature=random.uniform(0.8, 1.2),
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            streamer=text_streamer 
        )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # 3. Extract and Clean
        # Regex to find the last occurrence of Title/Text pattern
        matches = list(re.finditer(r"Title:\s*(.*?)\s*Text:\s*(.*?)(?=Title:|$)", generated_text, re.DOTALL))
        if not matches:
            # Fallback regex
            matches = list(re.finditer(r"Title:\s*(.*?)\s*Text:\s*(.*)", generated_text, re.DOTALL))
            
        if matches:
            match = matches[-1]
            title, text = match.groups()
            title = title.strip().strip('"')
            text = text.strip().strip('"').replace("\n", " ")
            
            # Simple content hash to check for near-duplicates
            content_hash = hash(text[:100])
            
            # 4. Validate and Save
            if (title not in existing_titles and 
                content_hash not in existing_content_hashes and 
                len(text) > 200):
                
                existing_titles.add(title)
                existing_content_hashes.add(content_hash)
                
                # Metrics
                unique_words = len(set(text.lower().split()))
                total_words = len(text.split())
                diversity_score = unique_words / total_words if total_words > 0 else 0
                
                stories.append({
                    "title": title, 
                    "text": text, 
                    "theme": theme,
                    "structure": structure,
                    "diversity_score": round(diversity_score, 2),
                    "word_count": total_words,
                    "score": 0
                })
                
                print(f"\n[+] Story {len(stories)}/{n} generated: '{title}'")
                
                # Periodic save (checkpointing)
                if len(stories) % 5 == 0:
                    pd.DataFrame(stories).to_csv(output_file, index=False)
                    print(f"💾 Checkpoint saved to {output_file}")

    # Final Save
    df = pd.DataFrame(stories)
    df.to_csv(output_file, index=False)
    print(f"\n🎉 Generation complete! {len(stories)} stories saved to {output_file}")
    return df

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Model
    model, tokenizer = load_base_model()
    
    # 2. Generate Stories
    df_stories = generate_stories(
        model, 
        tokenizer, 
        n=N_STORIES, 
        output_file=OUTPUT_CSV
    )