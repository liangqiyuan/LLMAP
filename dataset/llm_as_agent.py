import re
import json
from difflib import get_close_matches
from transformers import AutoModelForCausalLM, AutoTokenizer 

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="balanced",)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def format_map_data(scenario_data):
    filtered_pois = []
    type_groups = {}

    for poi in scenario_data['pois']:
        if poi['Opening'] and poi['Opening'][0] and 'Closed' not in poi['Opening'][0]:
            poi_type = poi['Type']
            if poi_type not in type_groups:
                type_groups[poi_type] = []
            if len(type_groups[poi_type]) < 10:
                type_groups[poi_type].append(poi)

    for pois in type_groups.values():
        filtered_pois.extend(pois)

    map_info = "\n\nAvailable POIs:"
    for i, poi in enumerate(filtered_pois):
        map_info += f'''
    POI ID: {poi['Place ID']}:
    * Type: {poi['Type']}
    * Rating: {poi['Rating']} ({poi['Number Ratings']} reviews)
    * Coordinates: ({poi['Latitude']}, {poi['Longitude']})
    * Opening Hours: {poi['Opening'][0]}
    '''
    return map_info

def generate_route(instruction, scenario_data, LLM_model, LLM_tokenizer, CoT=True):
    system_prompt = '''
You are a route planning assistant. Your goal is to plan an optimal route based on the following objectives:

Primary Objectives:
1. Minimize the total route length/distance 
2. Maximize coverage of different POI types (select exactly one POI per required type)
3. Maximize the quality of visited POIs (based on ratings and number of ratings)
4. Balance between route efficiency and POI quality
5. Ensure compliance with time limits from instructions
6. Account for dependencies between POIs
7. Respect opening hours of recommended POIs

Visit Duration: shopping mall: 120 mins, supermarket: 30 mins, pharmacy: 15 mins, bank: 20 mins, library: 60 mins
Travel Speed: 30 km/h
Departure Time: 10:00 AM

Always output POI IDs as provided in the input data. Your output must strictly follow this format:
[POI ID, POI ID, POI ID]
'''

    system_prompt_CoT = '''
You are a route planning assistant. Your goal is to plan an optimal route based on the following objectives:

Primary Objectives:
1. Minimize the total route length/distance 
2. Maximize coverage of different POI types (select exactly one POI per required type)
3. Maximize the quality of visited POIs (based on ratings and number of ratings)
4. Balance between route efficiency and POI quality
5. Ensure compliance with time limits from instructions
6. Account for dependencies between POIs
7. Respect opening hours of recommended POIs

Visit Duration: shopping mall: 120 mins, supermarket: 30 mins, pharmacy: 15 mins, bank: 20 mins, library: 60 mins
Travel Speed: 30 km/h
Departure Time: 10:00 AM

Planning Process:
1. Analyze User Requirements:
  - Identify required POI types from human instruction
  - Note any specified preferences for particular types

2. Prioritize POI Types:
  - Order POI types based on:
    * User specified preferences/requirements
    * Dependencies between types
    * Opening hours

3. Select Specific POIs:
  - For each POI type in order:
    * Consider only POIs of that specific type
    * Choose exactly one POI based on:
      - Rating and number of reviews
      - Location efficiency (distance to previous/next points)
      - Opening hours compatibility
  - Ensure only one POI is selected per type

First analyze the constraints and requirements, then plan accordingly. After planning, validate that your route satisfies all constraints.

Always output POI IDs as provided in the input data. Your output must strictly follow this format:
[POI ID, POI ID, POI ID]
'''
    messages = f"<|system|>{system_prompt_CoT}\n\n{format_map_data(scenario_data)}\n\n" if CoT else f"<|system|>{system_prompt}\n\n{format_map_data(scenario_data)}\n\n"
    messages += f"<|user|>{instruction}\n\nOnly output the list object. No prefix, additional text, or explanation."
    messages += "<|assistant|>"
    inputs = LLM_tokenizer(messages, return_tensors="pt")
    outputs = LLM_model.generate(inputs['input_ids'], max_new_tokens=256, pad_token_id=LLM_tokenizer.eos_token_id)
    text = LLM_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response = text[text.rfind("<|assistant|>") + len("<|assistant|>"):].strip()
    return response

def generate_route_in_sample(model_names, llm_agent_dataset):
    for CoT in [False, True]:
        for model_name in model_names:
            LLM_model, LLM_tokenizer = load_model(model_name)
            for sample in llm_agent_dataset:
                key = model_name + '_CoT' if CoT else model_name
                if key in sample['llm_agent']:
                    continue

                route = generate_route(sample['human_instruction'], sample['scenario'], LLM_model, LLM_tokenizer, CoT)
                result = re.findall(r'ChIJ[a-zA-Z0-9_-]{23}', ''.join(route))
                valid_pois = [poi['Place ID'] for poi in sample['scenario']['pois']]
                matched_result = [match[0] for id in result if (match := get_close_matches(id, valid_pois, n=1, cutoff=0.5))]
                sample['llm_agent'][key] = matched_result

            with open('llm_agent_dataset.json', 'w', encoding='utf-8') as f:
                json.dump(llm_agent_dataset, f, ensure_ascii=False, indent=4)


model_names = [
    "microsoft/Phi-3.5-mini-instruct", 
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
]

if __name__ == "__main__":
    with open('llm_agent_dataset.json', 'r', encoding='utf-8') as f:
        llm_agent_dataset = json.load(f)
    generate_route_in_sample(model_names, llm_agent_dataset)
