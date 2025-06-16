"""Formalization Target wire
Wires:
1. W is th eset of all wires W subset R^d, w = {w_0, w_1, ..., w_n}
2. C is a set of possible colors C = {red, blue, black, white, orange, yellow, green}
    for any c in C, w_c = {w in W | color(w)=c}
3. c^* in C and denotes the target color
    w_c^* subset of W 
4. W^* in W_c^* is the target wire, we will define as the closest to the target terminal

Terminals:
1. P is the finite set of all terminals P subset R^d, each p_i in R^d
    P = {p_0, p_1, ..., p_8}
2. let e^* in {0, 1, ..., 8} be the index of the target terminal
3. p^* = p_t^* in P

Algorithm for defining target wire:
1. DistanceFunciton(w, p^*) =||pos(w) - pos(p^*)||
2. w^* = argmmin_w in W_c^*(DistanceFunction(w,p^*))

Action Map function maps to next action based on target wire's state
"""
import omni
from robot_help import find_position
import numpy as np
import math

class DeterministicActionModel:
    def __init__(self, wires, terminals):
        self.wires = wires
        self.terminals = terminals
        self.manipulated_object = False
        self.target_wire_state = ""
        self.target_wire_color = ""
        self.target_terminal = None
        
    def distance_function(self, pos1, pos2):
        """Euclidean Distance"""
        return math.sqrt(sum((a-b)**2 for a, b in zip(pos1, pos2))) 
     
    def process_scene(self, llm_data):
        self.target_terminal = self.terminals[llm_data["target_terminal"]]
        terminal_coords = self.terminals[self.target_terminal]["coordinates"]
        w_cstar = [wire for wire in self.wires if wire["name"] == llm_data["target_wire"]]
        
        if len(w_cstar) > 1:
            target_wire = self.wire_map_function(w_cstar)
        else:
            target_wire = w_cstar[0]
            self.target_wire_state = target_wire["state"]
        
        manipulated_wire = None # Assume potential for holding non-target wire
        for wire in self.wires:
            if wire["ID"] != target_wire["ID"] and wire["state"] == "held":
                manipulated_wire = wire
                self.manipulated_object = True
                break
        
        return target_wire, self.target_terminal, manipulated_wire  
    
    def wire_map_function(self, tar_wires):
        distances = []
        for wire in tar_wires:
            w_coords = wire["coordinates"]
            t_coords = self.target_terminal["coordinates"]
            distances.append(self.distance_function(w_coords, t_coords))
        
        target_wire = tar_wires[distances.index(min(distances))]
        self.target_wire_state = target_wire["state"]
        
        return target_wire

    def action_map_function(self):
        if self.target_wire_state == 'held':
            return 'insert'
        elif self.target_wire_state == 'inserted':
            return 'lock'
        elif self.target_wire_state == 'on_table':
            if self.manipulated_object:
                return 'putdown'
            else:
                return 'pick'
        else:
            return 'unknown'
