import matplotlib.pyplot as plt
import json
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer 
device = 'cuda:0'
client = OpenAI(api_key='...')

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device, torch_dtype="auto", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer

def generate_human_instruction_gpt4o(synthetic_label):
    system_prompt = '''
You need to generate a natural human instruction by following these thinking steps:

1. First sentence: State all the POIs that need to be visited today:
   - Look at the list of POIs
   - Think about natural way to express visiting multiple POIs
   
2. Second sentence: State the return time if specified:
   - If specific time given, express as deadline
   - If no time limit, omit this part
   
3. Third sentence: Express POI rating vs route length preference indirectly:
   - If POI rating weight > 0.5: Express strong desire for high rating POIs
   - If route length weight > 0.5: Express urgency or need for efficiency
   - If balanced: Express desire for both reasonable rating and efficiency
   
4. Forth sentence: Express each dependency separately, one by one:
   - Convert each [A,B] pair into natural sequence requirement
   - Think about natural ways to express "must visit A before B"

Keep the language natural but always follow this structure. 
'''

    user_content = f'''
Based on the following information, generate a natural human instruction:

1. POIs to visit: {', '.join(synthetic_label['pois'])}

2. Return by: {synthetic_label['time_limit']}

3. Preference analysis:
- POI rating weight: {synthetic_label['quality_weight']}
- Route length weight: {synthetic_label['distance_weight']}

4. Dependencies to express: {synthetic_label['dependencies']}

Generate ONE natural instruction that includes all this information. No prefix, additional text, or explanation.
'''
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}]
    
    response = client.chat.completions.create(model='gpt-4o', messages=messages)
    return response.choices[0].message.content
    

def estimate_from_instruction(instruction, LLM_model, LLM_tokenizer, CoT=False):
    system_prompt = '''
Extract POIs, constraints, and weights from a human instruction. POIs should be a simple string array. Dependencies should be an empty list if no sequence requirements are mentioned. Weights should be between 0 and 1, and sum to 1. Extract them from language about importance of POI rating vs route length. 

Output must be valid JSON with structure:
{
    "pois": ["poi1", "poi2", ...],
    "time_limit": "HH:00" or "None",
    "dependencies": [["poi1", "poi2"], ...],
    "quality_weight": float,
    "distance_weight": float
}
'''
      
    system_prompt_CoT = '''
Extract POIs, constraints, and weights from human instruction through step-by-step reasoning:

1. POIs Analysis:
    - Look for POIs mentioned that need to be visited
    - Create list of unique POIs

2. Time Limit Analysis:
    - Search for any specific return time
    - Format as HH:00 or "None" if not specified

3. Dependencies Analysis:
    - Look for words indicating sequence (before, after, then, etc.)
    - Create pairs of [POI1, POI2] for each sequence requirement

4. Preference Analysis:
    - Look for language about POI quality/rating importance vs route efficiency
    - High POI rating emphasis (quality, best places, etc.) ->quality_weight should be large than 0.5
    - High route efficiency emphasis (quick, shortest, save time) -> distance_weight should be large than 0.5
    - Balanced language -> both weights around 0.5

Output must be valid JSON with structure:
{
    "pois": ["poi1", "poi2", ...],
    "time_limit": "HH:00" or "None",
    "dependencies": [["poi1", "poi2"], ...],
    "quality_weight": float,
    "distance_weight": float
}
'''
  
    user_content = f'''
Extract POIs, constraints, and weights from this instruction as JSON:
{instruction}

Only output the JSON object. No prefix, additional text, or explanation.
'''

    messages = f"<|system|>{system_prompt_CoT}\n" if CoT else f"<|system|>{system_prompt}\n"
    messages += f"<|user|>{user_content}\n"
    messages += "<|assistant|>"
    inputs = LLM_tokenizer(messages, return_tensors="pt").to(device)
    outputs = LLM_model.generate(inputs['input_ids'], max_new_tokens=256, pad_token_id=LLM_tokenizer.eos_token_id)
    text = LLM_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response = text[text.rfind("<|assistant|>") + len("<|assistant|>"):].strip()
    if "<|" in response:
        response = response[:response.find("<|")].strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        try:
            if "{" in response and "}" in response:
                last_open = response.rindex("{")
                last_close = response.rindex("}")
                response = response[last_open:last_close + 1]
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                print(response)
                return {"pois": [], "time_limit": "None", "dependencies": [], "quality_weight": 0.5, " distance_weight": 0.5}
        except json.JSONDecodeError:
            print(response)
            return {"pois": [], "time_limit": "None", "dependencies": [], "quality_weight": 0.5, " distance_weight": 0.5}
 
def gpt_estimate_from_instruction(instruction, model_name, CoT=False):
    system_prompt = '''
Extract POIs, constraints, and weights from a human instruction. POIs should be a simple string array. Dependencies should be an empty list if no sequence requirements are mentioned. Weights should be between 0 and 1, and sum to 1. Extract them from language about importance of POI rating vs route length. 

Output must be valid JSON with structure:
{
    "pois": ["poi1", "poi2", ...],
    "time_limit": "HH:00" or "None",
    "dependencies": [["poi1", "poi2"], ...],
    "quality_weight": float,
    "distance_weight": float
}
'''
      
    system_prompt_CoT = '''
Extract POIs, constraints, and weights from human instruction through step-by-step reasoning:

1. POIs Analysis:
    - Look for POIs mentioned that need to be visited
    - Create list of unique POIs

2. Time Limit Analysis:
    - Search for any specific return time
    - Format as HH:00 or "None" if not specified

3. Dependencies Analysis:
    - Look for words indicating sequence (before, after, then, etc.)
    - Create pairs of [POI1, POI2] for each sequence requirement

4. Preference Analysis:
    - Look for language about POI quality/rating importance vs route efficiency
    - High POI rating emphasis (quality, best places, etc.) ->quality_weight should be large than 0.5
    - High route efficiency emphasis (quick, shortest, save time) -> distance_weight should be large than 0.5
    - Balanced language -> both weights around 0.5

Output must be valid JSON with structure:
{
    "pois": ["poi1", "poi2", ...],
    "time_limit": "HH:00" or "None",
    "dependencies": [["poi1", "poi2"], ...],
    "quality_weight": float,
    "distance_weight": float
}
'''
  
    user_content = f'''
Extract POIs, constraints, and weights from this instruction as JSON:
{instruction}

Only output the JSON object. No prefix, additional text, or explanation.
'''

    system_prompt_wwo = system_prompt_CoT if CoT else system_prompt
    messages = [{"role": "user", "content": system_prompt_wwo},
                {"role": "user", "content": user_content}]
    response = client.chat.completions.create(model=model_name, messages=messages)
    response = response.choices[0].message.content
    print(response)

    if response is None:
        return {"pois": [], "time_limit": "None", "dependencies": [], "quality_weight": 0.5, " distance_weight": 0.5}
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        try:
            if "{" in response and "}" in response:
                last_open = response.rindex("{")
                last_close = response.rindex("}")
                response = response[last_open:last_close + 1]
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                print(response)
                return {"pois": [], "time_limit": "None", "dependencies": [], "quality_weight": 0.5, " distance_weight": 0.5}
        except json.JSONDecodeError:
            print(response)
            return {"pois": [], "time_limit": "None", "dependencies": [], "quality_weight": 0.5, " distance_weight": 0.5}
        

def generate_dataset(num_samples):
    all_pois = ['shopping mall', 'supermarket', 'pharmacy', 'bank', 'library']
    # generation_model = "meta-llama/Llama-3.2-3B-Instruct"
    # LLM_model, LLM_tokenizer = load_model(generation_model)
    dataset = []
    for _ in range(num_samples):
        num_pois = random.randint(1, 5)
        selected_pois = random.sample(all_pois, num_pois)
    if random.random() < 0.3:
       hour = random.randint(17, 24)
       time_limit = f"{hour:02d}:00"
    else:
       time_limit = "None"
    dependencies = []
    for i in range(len(selected_pois) - 1):
       if random.random() < 0.3:
           dependencies.append([selected_pois[i], selected_pois[i + 1]])
    
    quality_weight = round(random.random(), 1)
    distance_weight = round(1 - quality_weight, 1)
   
    synthetic_label = {"pois": selected_pois, "time_limit": time_limit, "dependencies": dependencies, "quality_weight": quality_weight, " distance_weight":  distance_weight}
    human_instruction = generate_human_instruction_gpt4o(synthetic_label)
    dataset.append({"synthetic_label": synthetic_label, "human_instruction": human_instruction})
    return dataset

def generate_estimations(estimation_models, dataset):
    for CoT in [False, True]:
        for model_name in estimation_models:
            key = model_name + '_CoT' if CoT else model_name
            if key in dataset[0]:
                continue
            LLM_model, LLM_tokenizer = load_model(model_name)
            for sample in dataset:
                estimation = estimate_from_instruction(sample["human_instruction"], LLM_model, LLM_tokenizer, CoT=CoT)
                sample[key] = estimation
                
            del LLM_model, LLM_tokenizer
            torch.cuda.empty_cache()

            with open('HIPP.json', 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)
    return dataset

def gpt_generate_estimations(estimation_models, dataset):
    for CoT in [False, True]:
        for model_name in estimation_models:
            key = model_name + '_CoT' if CoT else model_name
            if key in dataset[0]:
                continue

            print(model_name)
            for sample in dataset:
                estimation = gpt_estimate_from_instruction(sample["human_instruction"], model_name, CoT=CoT)
                sample[key] = estimation

            with open('HIPP.json', 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)
    return dataset


if __name__ == "__main__":

    human_instructions = generate_dataset(num_samples=1000)
  
    estimation_models = [
        "microsoft/Phi-3-mini-4K-instruct",
        "microsoft/Phi-3.5-mini-instruct", 
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
    ]

    estimation = generate_estimations(estimation_models, human_instruction)

    estimation_models = [
        'gpt-3.5-turbo-0125',
        'gpt-4o',
        'gpt-4o-mini', 
        'o1',
        'o1-mini'
    ]

    estimation = gpt_generate_estimations(estimation_models, human_instruction)

    
