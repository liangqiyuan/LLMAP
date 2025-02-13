import numpy as np
from tqdm import tqdm
import json
import pickle
import time
from haversine import haversine
import itertools
import re
import networkx as nx

search_type_mapping = {"shopping_mall": 0, "supermarket": 1, "pharmacy": 2, "bank": 3, "library": 4}
stay_time_mapping = {0: 120, 1: 30, 2: 15, 3: 20, 4: 60}
average_speed = 0.5 # km/min
distance_factor = 1.5

day_to_index = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
current_day = "Monday"
departure_time = 10

def filtered_poi(sample):
   all_pois = sample["scenario"]["pois"]
   processed_pois = []
   for poi in all_pois:
       if poi['Rating'] == 'N/A':
           poi['Rating'] = 1.0
       if poi['Number Ratings'] == 'N/A':
           poi['Number Ratings'] = 1
       if not poi.get('Opening') or any(time == '' for time in poi['Opening']):
           poi['Opening'] = ["Monday: 9:00 AM – 5:00 PM"]
       processed_pois.append(poi)
   return processed_pois

def LLMAP(dataset_file):
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    output = {}
    synthetic_graphs = []
    method_graphs = {method: [] for method in dataset[0]["llm_estimation"].keys()}
    
    for sample in tqdm(dataset, desc="Construct Graph"):
        synthetic_pois = set(sample["synthetic_label"]['pois'])
        all_pois = filtered_poi(sample)
        
        base_filtered_pois = [poi for poi in all_pois if poi['Type'] in synthetic_pois]
        
        num_nodes = len(base_filtered_pois) + 2
        nodes = list(range(num_nodes))
        positions = {}
        groups = [-1] * num_nodes
        start_node = 0
        goal_node = num_nodes - 1
        
        positions[start_node] = (sample['scenario']['start_location']['latitude'], sample['scenario']['start_location']['longitude'])
        positions[goal_node] = (sample['scenario']['end_location']['latitude'], sample['scenario']['end_location']['longitude'])

        base_node_mapping = {}
        for idx, poi in enumerate(base_filtered_pois, start=1):
            positions[idx] = (poi['Latitude'], poi['Longitude'])
            poi_type = search_type_mapping[poi['Type']]
            groups[idx] = poi_type
            base_node_mapping[poi['Type']] = idx

        base_positions = [positions[node] for node in nodes]
        
        base_edge_indices = []
        base_edge_weights = []
        for i in nodes:
            for j in nodes:
                if i != j and groups[i] != groups[j]:
                    base_edge_indices.append([i, j])
                    dist = haversine(base_positions[i], base_positions[j])
                    base_edge_weights.append(dist)

        if [start_node, goal_node] not in base_edge_indices:
            base_edge_indices.append([start_node, goal_node])
            dist = haversine(base_positions[start_node], base_positions[goal_node])
            base_edge_weights.append(dist)

        base_graph = {
            'positions': base_positions,
            'edge_indices': base_edge_indices,
            'edge_weights': base_edge_weights,
            'start_node': start_node,
            'goal_node': goal_node,
            'groups': groups.copy(),
            'time_constraint': sample["synthetic_label"]['time_constraint'],
            'ratings': [-1.0] + [float(poi['Rating']) for poi in base_filtered_pois] + [-1.0],
            'num_ratings': [-1] + [poi['Number Ratings'] for poi in base_filtered_pois] + [-1],
            'dependencies': sample["synthetic_label"]['dependencies'],
            'openings': [''] + [poi['Opening'] for poi in base_filtered_pois] + [''],
            'alpha': sample["synthetic_label"]['rating_weight'] / 2,
            'beta': sample["synthetic_label"]['route_weight']
        }
        synthetic_graphs.append(base_graph)
        
        for method in sample["llm_estimation"]:
            method_pois = set(sample["llm_estimation"][method].get('pois', []))
            
            if method_pois == synthetic_pois:
                method_graph = base_graph.copy()
                method_graph.update({
                    'time_constraint': sample["llm_estimation"][method].get('time_constraint', None),
                    'dependencies': sample["llm_estimation"][method].get('dependencies', []),
                    'alpha': sample["llm_estimation"][method].get('rating_weight', 0.5) / 2,
                    'beta': sample["llm_estimation"][method].get('route_weight', 0.5)
                })
            else:
                method_filtered_pois = [poi for poi in all_pois if poi['Type'] in method_pois]
                new_num_nodes = len(method_filtered_pois) + 2
                new_positions = {0: positions[0], new_num_nodes-1: positions[goal_node]}
                new_groups = [-1] * new_num_nodes
                
                for idx, poi in enumerate(method_filtered_pois, start=1):
                    new_positions[idx] = (poi['Latitude'], poi['Longitude'])
                    new_groups[idx] = search_type_mapping[poi['Type']]
                
                new_positions_list = [new_positions[node] for node in range(new_num_nodes)]
                
                new_edge_indices = []
                new_edge_weights = []
                for i in range(new_num_nodes):
                    for j in range(new_num_nodes):
                        if i != j and new_groups[i] != new_groups[j]:
                            new_edge_indices.append([i, j])
                            dist = haversine(new_positions_list[i], new_positions_list[j])
                            new_edge_weights.append(dist)

                if [0, new_num_nodes-1] not in new_edge_indices:
                    new_edge_indices.append([0, new_num_nodes-1])
                    dist = haversine(new_positions_list[0], new_positions_list[new_num_nodes-1])
                    new_edge_weights.append(dist)
                
                method_graph = {
                    'positions': new_positions_list,
                    'edge_indices': new_edge_indices,
                    'edge_weights': new_edge_weights,
                    'start_node': 0,
                    'goal_node': new_num_nodes - 1,
                    'groups': new_groups,
                    'time_constraint': sample["llm_estimation"][method].get('time_constraint', None),
                    'ratings': [-1.0] + [float(poi['Rating']) for poi in method_filtered_pois] + [-1.0],
                    'num_ratings': [-1] + [poi['Number Ratings'] for poi in method_filtered_pois] + [-1],
                    'dependencies': sample["llm_estimation"][method].get('dependencies', []),
                    'openings': [''] + [poi['Opening'] for poi in method_filtered_pois] + [''],
                    'alpha': sample["llm_estimation"][method].get('rating_weight', 0.5) / 2,
                    'beta': sample["llm_estimation"][method].get('route_weight', 0.5)
                }
            method_graphs[method].append(method_graph)
    
    st = time.time()
    synthetic_paths, synthetic_groups = MSGD(synthetic_graphs)
    synthetic_runtime = time.time() - st
    output['synthetic_label'] = {
        'graphs': synthetic_graphs,
        'paths': synthetic_paths,
        'groups': synthetic_groups,
        'total_runtime': synthetic_runtime
    }
    
    for method in tqdm(method_graphs, desc="MSGD"):
        st = time.time()
        method_paths, method_groups = MSGD(method_graphs[method])
        method_runtime = time.time() - st
        output[f'{method}'] = {
            'graphs': method_graphs[method],
            'paths': method_paths,
            'groups': method_groups,
            'total_runtime': method_runtime
        }
      
    return output




def compute_path_and_cost(i, graph, selected_groups, node_in_groups):
    start_node = graph[i]['start_node']
    goal_node = graph[i]['goal_node']
    positions = graph[i]['positions']
    groups = graph[i]['groups']
    ratings = graph[i]['ratings']
    num_ratings = graph[i]['num_ratings']
    openings = graph[i]['openings']
    alpha = graph[i]['alpha']
    beta = graph[i]['beta']

    rating_range = (1, 5)
    stay_time_range = (15, 120)
    travel_times = [haversine(positions[n1], positions[n2]) * distance_factor / average_speed / 60 for group in node_in_groups for n1 in group for n2 in group if n1 != n2]

    num_ratings_list = [num_ratings[node] for group in node_in_groups for node in group]
    travel_time_range = (min(travel_times), max(travel_times))
    num_ratings_range = (min(num_ratings_list), max(num_ratings_list))
    
    def min_max_normalize(value, value_range, zero_case=0.0):
        if value_range[0] == value_range[1]:
            return zero_case
        return (value - value_range[0]) / (value_range[1] - value_range[0])

    def calculate_real_time(node1, node2):
        travel_time = haversine(positions[node1], positions[node2]) * distance_factor / average_speed / 60
        stay_time = stay_time_mapping[groups[node2]] / 60 if groups[node2] != -1 else 0
        return travel_time + stay_time

    def calculate_weight(node1, node2):
        travel_time = min_max_normalize(haversine(positions[node1], positions[node2]) * distance_factor / average_speed / 60, travel_time_range, zero_case=0.0)
        stay_time = min_max_normalize(stay_time_mapping[groups[node2]] / 60 if groups[node2] != -1 else 0, stay_time_range, zero_case=0.0)
        if node2 == goal_node:
            return travel_time + stay_time
        node_rating = min_max_normalize(ratings[node2], rating_range, zero_case=1.0)
        node_num_ratings = min_max_normalize(num_ratings[node2], num_ratings_range, zero_case=1.0)
        return - (alpha * node_rating + alpha * node_num_ratings) + beta * (travel_time + stay_time)
    
    edges = [(start_node, node, calculate_weight(start_node, node)) 
            for node in node_in_groups[selected_groups[0]]]
    edges.extend([(prev_node, next_node, calculate_weight(prev_node, next_node))
                 for prev_group, next_group in zip(selected_groups[:-1], selected_groups[1:])
                 for prev_node in node_in_groups[prev_group]
                 for next_node in node_in_groups[next_group]])
    edges.extend([(node, goal_node, calculate_weight(node, goal_node)) 
                 for node in node_in_groups[selected_groups[-1]]])
    
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges)
    _, path = nx.single_source_dijkstra(G, source=start_node, target=goal_node, weight='weight')
    if not check_path_availability(path, groups, goal_node, positions, openings):
        return float('inf'), path

    real_time = departure_time + sum(calculate_real_time(path[i], path[i+1]) for i in range(len(path)-1))
    return real_time, path


def MSGD(graphs):
    predicted_paths = []
    predicted_groups = []

    for i in range(len(graphs)):
        start_node = graphs[i]['start_node']
        goal_node = graphs[i]['goal_node']
        groups = graphs[i]['groups']
        time_constraint = graphs[i]['time_constraint']
        time_constraint = float('inf') if time_constraint is None else (float(time_constraint.split(':')[0]) if time_constraint.split(':')[0].isdigit() else float('inf'))
        dependencies = graphs[i]['dependencies']
        node_in_groups = []

        group_indices = np.unique([group_id for group_id in groups if group_id >= 0]).tolist()
        for group_id in group_indices:
            node_indices = np.where(np.array(groups) == group_id)[0]
            node_in_group = [idx for idx in node_indices if idx not in [start_node, goal_node]]
            node_in_groups.append(node_in_group)
        group_indices = list(range(len(node_in_groups)))

        def is_valid_order(order):
            for dependency in dependencies:
                if len(dependency) != 2:
                    continue
                if dependency[0] in search_type_mapping and dependency[1] in search_type_mapping:
                    if search_type_mapping[dependency[0]] in order and search_type_mapping[dependency[1]] in order:
                        if order.index(search_type_mapping[dependency[0]]) > order.index(search_type_mapping[dependency[1]]):
                            return False
            return True

        best_weight = float('inf')
        best_path = None
        best_order = None
        for num_groups in range(len(group_indices), 0, -1):
            for groups_subset in itertools.combinations(group_indices, num_groups):
                for perm in itertools.permutations(groups_subset):
                    if is_valid_order(perm):
                        w, p = compute_path_and_cost(i, graphs, perm, node_in_groups)
                        if w <= time_constraint and w < best_weight:
                            best_weight = w
                            best_path = p
                            best_order = perm
                            predicted_paths.append(best_path)
                            predicted_groups.append(list(best_order))
                            break
                if best_path is not None:
                    break
            if best_path is not None:
                break
               
        if best_path is None:
            predicted_paths.append([start_node, goal_node])
            predicted_groups.append([])

    return predicted_paths, predicted_groups

def parse_time(time_str):
    if time_str.endswith("Closed"):
        return []
    if "Open 24 hours" in time_str:
        return [(0.0, 24.0)]
    time_ranges = []
    for time_range in time_str.split(","):
        time_range = time_range.strip()
        if "–" not in time_range:
            continue
        start_time_str, end_time_str = time_range.split("–")
        pattern = r"(\d+):(\d+)\s*(AM|PM)"
        start_match = re.search(pattern, start_time_str)
        end_match = re.search(pattern, end_time_str)
        if not (start_match and end_match):
            continue
            
        def convert_to_hours(match):
            hours = int(match.group(1))
            minutes = int(match.group(2))
            if match.group(3) == "PM" and hours != 12:
                hours += 12
            elif match.group(3) == "AM" and hours == 12:
                hours = 0
            return hours + minutes / 60
            
        time_ranges.append((convert_to_hours(start_match), convert_to_hours(end_match)))
    return time_ranges

def check_path_availability(path, groups, goal_node, positions, opening_hours, start_time=departure_time):
    current_time = start_time
    for i in range(len(path)-1):
        node1, node2 = path[i], path[i+1]
        travel_time = haversine(positions[node1], positions[node2]) * distance_factor / average_speed / 60
        current_time += travel_time
        if node2 != goal_node:
            day_opening_hours = opening_hours[node2][day_to_index[current_day]]

            def is_available(current_time, opening_times):
                for start_time, end_time in opening_times:
                    if start_time <= current_time <= end_time:
                        return True
                return False
            
            if not is_available(current_time, parse_time(day_opening_hours)):
                return False
            current_time += stay_time_mapping[groups[node2]] / 60 if groups[node2] != -1 else 0
    return True





















def evaluate_llm_parser_paths(dataset):
    results = {
        'ratings': [],
        'num_ratings': [],
        'path_length': [],
        'group_coverage': [],
        'time_violations': [],
        'dependency_violations': [],
        'availability_violations': []
    }

    predicted_paths = dataset['paths']
    predicted_groups = dataset['groups']
    graphs = dataset['graphs']

    for graph, predicted_path, predicted_group_order in zip(graphs, predicted_paths, predicted_groups):
        goal_node = graph['goal_node']
        positions = graph['positions']
        edge_indices = graph['edge_indices']
        edge_weights = graph['edge_weights']
        groups = graph['groups']
        ratings = graph['ratings']
        num_ratings = graph['num_ratings']
        time_constraint = graph['time_constraint']
        time_constraint = float('inf') if time_constraint is None else (float(time_constraint.split(':')[0]) if time_constraint.split(':')[0].isdigit() else float('inf'))
        dependencies = graph['dependencies']
        openings = graph['openings']

        covered_groups = set([groups[node] for node in predicted_path])
        total_groups = set(np.unique([group_id for group_id in groups if group_id >= 0]).tolist())
        covered_groups.discard(-1)
        total_groups.discard(-1)

        valid_covered_groups = covered_groups.intersection(total_groups)
        group_coverage_rate = len(valid_covered_groups) / len(total_groups) if len(total_groups) > 0 else 0
        results['group_coverage'].append(group_coverage_rate * 100)

        path_ratings = []
        path_num_ratings = []
        path_length = 0

        for node in predicted_path[1:-1]:
            path_ratings.append(ratings[node])
            path_num_ratings.append(num_ratings[node])

        results['ratings'].append(sum(path_ratings) / len(path_ratings) if path_ratings else 0)
        results['num_ratings'].append(sum(path_num_ratings) / len(path_num_ratings) if path_num_ratings else 0)

        edge_indices = np.array(edge_indices)
        for u, v in zip(predicted_path[:-1], predicted_path[1:]):
            mask = (edge_indices[:, 0] == u) & (edge_indices[:, 1] == v)
            idx = np.where(mask)[0][0]
            path_length += edge_weights[idx] * distance_factor

        results['path_length'].append(path_length)

        def calculate_real_time(node1, node2):
            travel_time = haversine(positions[node1], positions[node2]) * distance_factor / average_speed / 60
            stay_time = stay_time_mapping[groups[node2]] / 60 if groups[node2] != -1 else 0
            return travel_time + stay_time

        real_time = departure_time + sum(calculate_real_time(predicted_path[i], predicted_path[i+1]) for i in range(len(predicted_path)-1))
        results['time_violations'].append(max(0, real_time - time_constraint))


        for dep in dependencies:
            if len(dep) != 2:
                continue
            if dep[0] in search_type_mapping and dep[1] in search_type_mapping:
                if search_type_mapping[dep[0]] in predicted_group_order and search_type_mapping[dep[1]] in predicted_group_order:
                    if predicted_group_order.index(search_type_mapping[dep[0]]) > predicted_group_order.index(search_type_mapping[dep[1]]):
                        results['dependency_violations'].append(1)
                        break
        else:
            results['dependency_violations'].append(0)

        if not check_path_availability(predicted_path, groups, goal_node, positions, openings):
            results['availability_violations'].append(1)
        else:
            results['availability_violations'].append(0)

    return results

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



def evaluate_llm_agent_paths(dataset, method):
    results = {
        'ratings': [],
        'num_ratings': [],
        'path_length': [],
        'group_coverage': [],
        'time_violations': [],
        'dependency_violations': [],
        'availability_violations': []
    }
    
    for sample in dataset:
        all_pois = {poi['Place ID']: poi for poi in sample['scenario']['pois']}
        place_ids = sample['llm_agent'][method]
        
        path_ratings = []
        path_num_ratings = []
        positions = []
        groups = []
        groups_in_path = set()
        total_groups = set()
        openings = []
        
        start_pos = (float(sample['scenario']['start_location']['latitude']), float(sample['scenario']['start_location']['longitude']))
        positions.append(start_pos)
        groups.append(-1)
        openings.append('')
        
        for place_id in place_ids:
            poi = all_pois[place_id]
            
            rating = float(poi['Rating']) if poi['Rating'] != 'N/A' else 1.0
            num_rating = int(poi['Number Ratings']) if poi['Number Ratings'] != 'N/A' else 1
            
            path_ratings.append(rating)
            path_num_ratings.append(num_rating)
            positions.append((float(poi['Latitude']), float(poi['Longitude'])))
            groups_in_path.add(search_type_mapping[poi['Type']])
            total_groups.add(search_type_mapping[poi['Type']])
            groups.append(search_type_mapping[poi['Type']])
            openings.append(poi.get('Opening', []) if poi.get('Opening') and any(poi['Opening']) else ["Monday: 9:00 AM – 5:00 PM"])
        
        end_pos = (float(sample['scenario']['end_location']['latitude']), float(sample['scenario']['end_location']['longitude']))
        positions.append(end_pos)
        groups.append(-1)
        openings.append('')
        
        results['ratings'].append(sum(path_ratings) / len(path_ratings) if path_ratings else 0)
        results['num_ratings'].append(sum(path_num_ratings) / len(path_num_ratings) if path_num_ratings else 0)
        
        path_length = 0
        for i in range(len(positions)-1):
            path_length += haversine(positions[i], positions[i+1]) * distance_factor
        results['path_length'].append(path_length)

        for poi in sample['scenario']['pois']:
            total_groups.add(search_type_mapping[poi['Type']])
        group_coverage = len(groups_in_path) / len(total_groups) if total_groups else 0
        results['group_coverage'].append(group_coverage * 100)

        def calculate_real_time(pos1, pos2, group2):
            travel_time = haversine(pos1, pos2) * distance_factor / average_speed / 60
            stay_time = stay_time_mapping[group2] / 60 if group2 != -1 else 0
            return travel_time + stay_time

        time_constraint = sample['synthetic_label']['time_constraint']
        time_constraint = float('inf') if time_constraint is None else (float(time_constraint.split(':')[0]) if time_constraint.split(':')[0].isdigit() else float('inf'))
        real_time = departure_time + sum(calculate_real_time(positions[i], positions[i+1], groups[i+1]) for i in range(len(positions)-1))
        results['time_violations'].append(max(0, real_time - time_constraint))

        dependencies = sample['synthetic_label']['dependencies']
        for dep in dependencies:
            if len(dep) != 2:
                continue
            if dep[0] in search_type_mapping and dep[1] in search_type_mapping:
                if search_type_mapping[dep[0]] in groups and search_type_mapping[dep[1]] in groups:
                    if groups.index(search_type_mapping[dep[0]]) > groups.index(search_type_mapping[dep[1]]):
                        results['dependency_violations'].append(1)
                        break
        else:
            results['dependency_violations'].append(0)

        if not check_path_availability(list(range(len(positions))), groups, len(positions)-1, positions, openings):
            results['availability_violations'].append(1)
        else:
            results['availability_violations'].append(0)

    return results

def evaluate_all_agent_methods():
    with open("final_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    all_results = {}
    for method in model_map.keys():
        results = evaluate_llm_agent_paths(dataset, method)
        all_results[method] = results
    
    output_file = "llm_agent_evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        
    return all_results
