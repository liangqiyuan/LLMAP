from utils import *

model_map = {
    "synthetic_label": "Synthetic Label",
    "microsoft/Phi-3-mini-4K-instruct": "Phi-3-mini",
    "microsoft/Phi-3.5-mini-instruct": "Phi-3.5-mini", 
    "meta-llama/Llama-3.2-3B-Instruct": "LLaMA-3.2-3B",
    "meta-llama/Llama-3.1-8B-Instruct": "LLaMA-3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B", 
    "google/gemma-2-2b-it": "Gemma-2-2B",
    "google/gemma-2-9b-it": "Gemma-2-9B",
    "openai/gpt-3.5-turbo-0125": "GPT-3.5",
    "openai/gpt-4o": "GPT-4o",
    "openai/gpt-4o-mini": "GPT-4o-mini",
    "openai/o1-mini": "OpenAI o1-mini",
    "openai/o1": "OpenAI o1",
   
    "microsoft/Phi-3-mini-4K-instruct_CoT": "Phi-3-mini \CoT",
    "microsoft/Phi-3.5-mini-instruct_CoT": "Phi-3.5-mini \CoT",
    "meta-llama/Llama-3.2-3B-Instruct_CoT": "LLaMA-3.2-3B \CoT", 
    "meta-llama/Llama-3.1-8B-Instruct_CoT": "LLaMA-3.1-8B \CoT",
    "mistralai/Mistral-7B-Instruct-v0.3_CoT": "Mistral-7B \CoT",
    "google/gemma-2-2b-it_CoT": "Gemma-2-2B \CoT", 
    "google/gemma-2-9b-it_CoT": "Gemma-2-9B \CoT",
    "openai/gpt-3.5-turbo-0125_CoT": "GPT-3.5 \CoT",
    "openai/gpt-4o-mini_CoT": "GPT-4o-mini \CoT",
    "openai/gpt-4o_CoT": "GPT-4o \CoT",
    "openai/o1-mini_CoT": "OpenAI o1-mini \CoT",
    "openai/o1_CoT": "OpenAI o1 \CoT",
}


dataset_file = "..."
dataset = LLMAP(dataset_file)

def evaluate_all_llm_parser():
    with open('llm_parser_data.pkl', 'rb') as f:
        dataset = pickle.load(f)
        
    all_results = {}
    for method in model_map.keys():
        results = evaluate_llm_parser_paths(dataset[method])
        all_results[method] = results
    
    output_file = "llm_parser_evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        
    return all_results

all_results = evaluate_all_llm_parser()













