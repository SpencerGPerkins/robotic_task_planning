"""
Pseudo-code
    while task not done:
        1. llm_response(input_prompt) -> task
        2. process_task(task) -> target wire ID, target terminal ID <task>
        3. retrieve_scene -> scene
        4. process_scene(scene) -> 
        4. ActionMap(target wire, manipulated wire) -> action
        5. Execute(action, target wire, terminal)
        6. Update() -> new states <scene>, new coords <scene>
        7. Task done?
"""
import numpy as np
import omni.timeline
import omni.ui as ui
import asyncio
import os
from pxr import Gf, Usd
from .function import pickup, insert, go_home_1,go_home_2, lock, putdown
from .robot_help import screenshot_with_depth, go_to_random_point, find_position, partial_reset, find_wire
from .sim_vision_data import SimVisionData
import string
import base64
import asyncio
import re
import json
from .LLM2 import generate_response
from deterministic_model import DeterministicModel
import math

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a-b)**2 for a, b in zip(p1,p2)))

def process_task(rsp):
    llm_words = rsp.split("-")
    print(f"\nLLM Words: {llm_words}\n")
    end_variants = ["lock", "(lock", "(locked", "(install", "install"]
    pick_variants = ["pick", "pickup", "(holding", "hold"]
    insert_variants = ["insert", "(insert", "inserted" ,"(inserted"]
    
    for word in llm_words:
        word = word.lower()
        if word in end_variants:
            goal = "lock"
        elif word in pick_variants:
            goal = "pick"
        elif word in insert_variants:
            goal = "insert"
    print(f"Goal Processed: goal is {goal}")
    
    # Remove all punctuation except '-' and '_'
    punct_to_remove = string.punctuation.replace('-', '').replace('_', '')
    trans_table = str.maketrans('', '', punct_to_remove + string.whitespace)           
    response = response.translate(trans_table)
    print(f"Text response: {response}")            
    
    if goal != "pick":
        _, llm_target_wire, llm_target_terminal = response.split("-")
    elif goal == "pick":
        _, llm_target_wire = response.split("-")
        llm_target_terminal = "terminal_0" # Arbitrarily assign terminal
        
    llm_out_data = {
        "goal": goal,
        "target_wire": llm_target_wire,
        "target_terminal": llm_target_terminal
    }       
    
    return llm_out_data, goal

def process_scene(wires, terminals, llm_data):
    target_terminal = llm_data["target_terminal"]
    terminal_coords = terminals[target_terminal]["coordinates"]
    matching_target_wires = [wire for wire in wires if wire["name"] == llm_data["target_wire"]]
    
    if len(matching_target_wires) > 1:
        matching_target_wires.sort(key=lambda wire: euclidean_distance(wire["coordinates"], terminal_coords))
    
    target_wire = matching_target_wires[0]
    
    manipulated_wire = None # Assume potential for holding non-target wire
    for wire in wires:
        if wire["ID"] != target_wire["ID"] and wire["state"] == "held":
            manipulated_wire = wire
            break
    
    return target_wire, target_terminal, manipulated_wire  

def make_json_serializable(obj):
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    else:
        return obj
    
def find_wire(wire_in):
    stage = omni.usd.get_context().get_stage()
    for w, prim_wire in enumerate(stage.Traverse()):
        prim_wire_name = prim_wire.GetName()
        if prim_wire_name == wire_in["name"]:
            wire_position = find_position(prim_wire.GetPath())
            wire_position = make_json_serializable(wire_position)
    return wire_position

def main():
    data = SimVisionData() 
    data.generate_file() # Initialize scene, function will save as json file (/vision/sim_vis_to_gnn.json)
    
    # Outer instructionloop (user prompt + initial processing)
    while True:
        task_done = False # Inner loop conditional
        user_input = input("Give Prompt (exit to quit): ", )
        if user_input.lower() == "exit":
            break
        # Save data for history and next action
        data_saver = {
            "predicted_action":[],
            "predicted_wire": [],
            "predicted_terminal":[],
            "previous_actions":[],
            "previous_wire":[]
        }
        with open("/home/ai/Desktop/simulation/extension/newstart_python/run_data/action_executions.json", "r") as in_file:
            history = json.load(in_file)

        data_saver["previous_actions"] = history["previous_actions"]
        data_saver["previous_wire"] = history["previous_wire"]
       
        while not task_done: # Task level loop
            # Gather directory info for data sample file name 
            lst = os.listdir("/home/ai/Desktop/simulation/extension/newstart_python/synthetic_data/llm/") # your directory path
            number_files = len(lst)
            print(f"Synthetic_data Sanity Check: Number of Files = {number_files}")
                                 
            response = generate_response(user_input, 1) 
            processed_llm_data, goal = process_task()
            
            # Save llm data sample
            with open(f"/home/ai/Desktop/simulation/extension/newstart_python/synthetic_data/llm/sample_{number_files}.json", "w") as llm_out: 
                json.dump(processed_llm_data, llm_out)
                print(f"SYNTHETIC DATA SAVED (filename=sample_{number_files}.json)")
                
            # Load vision file for mapping predicted wire ID
            with open("/home/ai/Desktop/simulation/extension/newstart_python/run_data/vision/sim_vis_to_gnn.json", "r") as vision:
                vision_dict = json.load(vision)
                
            # Save vision data sample
            with open(f"/home/ai/Desktop/simulation/extension/newstart_python/synthetic_data/vision/sample_{number_files}.json", "w") as out_file:
                json.dump(vision_dict, out_file)
            
            wires = vision_dict["wires"] # list of dicts per wire 
            terminals = vision_dict["terminals"] # dict if dicts per terminal (terminal name is key (e.g. terminal_0))

            true_target_wire, true_target_terminal, held_object = process_scene(wires, terminals, processed_llm_data)
            
            model = DeterministicModel(true_target_wire, held_object)
            action = model.map_funciton()
            
            data_saver["predicted_action"] = [action]
            data_saver["predicted_wire"] = true_target_wire["ID"]
            data_saver["predicted_terminal"] = true_target_terminal["name"]
            data_saver["previous_actions"].append(action)
            data_saver["previous_wire"].append(true_target_wire["ID"])
            with open("/home/ai/Desktop/simulation/extension/newstart_python/run_data/action_executions.json", "w") as file:
                json.dump(data_saver, file)       
                
            # Get xyz either from 3 or 6 dim coords
            if len(true_target_wire["coordinates"]) == 2:
                wire_coords = true_target_wire["coordinates"][0]
            else: 
                wire_coords = true_target_wire["coordinates"]
            if len(true_target_terminal["coordinates"]) == 2:
                terminal_coords = terminals[true_target_terminal]["coordinates"] [0]
            else: 
                terminal_coords = terminals[true_target_terminal]["coordinates"]    
                
            label_info = {
                    "target_wire": {
                        "ID": true_target_wire["ID"],
                        "name": true_target_wire["name"],
                        "color": true_target_wire["color"],
                        "coordinates": [wire_coords]
                    },
                    "target_terminal": {
                        "name": true_target_terminal,
                        "coordinates": [terminal_coords]
                    },
                    "correct_action": action,
                }       
                 
            # Save Label Synthetic Data
            with open(f"/home/ai/Desktop/simulation/extension/newstart_python/synthetic_data/labels/sample_{number_files}.json", "w") as label_out: # Synthetic data File
                json.dump(label_info, label_out)
            
            await screenshot_with_depth(filename=f"sample_{number_files}")

            if action == "pick":
                await pickup(true_target_wire["name"])
                vision_dict["wires"][true_target_wire["ID"]]["state"] = "held"
                vision_dict["wires"][true_target_wire["ID"]]["coordinates"] = find_wire(true_target_wire["name"])
                if goal == "pick":
                    task_done = True
            elif action == "insert":
                await insert(true_target_terminal)
                vision_dict["wires"][true_target_wire["ID"]]["state"] = "inserted"
                vision_dict["wires"][true_target_wire["ID"]]["coordinates"] = find_wire(true_target_wire["name"])
                vision_dict["terminals"][true_target_terminal]["state"] = "inserted"
                if goal == "insert":
                    task_done = True
            elif action == "lock":
                await lock(true_target_wire["name"], true_target_terminal)
                vision_dict["terminals"][true_target_terminal]["state"] = "locked"
                if goal == "lock":
                    task_done = True
            elif action == "putdown":
                await putdown()
                for wire in wires:
                    if wire["ID"] == history["previous_wire"][-1]:
                        wire["state"] = "on_table"
                    vision_dict["wires"][wire["ID"]]["coordinates"] = find_wire(wire["name"])

            # Save the updated dictionary
            with open("/home/ai/Desktop/simulation/extension/newstart_python/run_data/vision/sim_vis_to_gnn.json", "w") as overwrite:
                json.dump(vision_dict, overwrite, indent=2)
            
            
                            
                        
            
            
            
            
            
                
                                      
             
            
            

                
                                        

