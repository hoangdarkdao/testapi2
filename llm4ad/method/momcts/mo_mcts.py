from __future__ import annotations
import random
import copy
from pymoo.indicators.hv import Hypervolume
import math
import numpy as np
from typing import List, Tuple, Any, Optional  # Ensure Optional is imported
import time
import json
import os
def serialize_pareto_front(pareto_front: list[list[float, float]]):
    """
    Converts the global Pareto front (list of tuples) into a JSON-serializable list of dictionaries.
    """
    serialized_data = []
    for reward_vector in pareto_front:
        if isinstance(reward_vector, np.ndarray):
            reward_vector = reward_vector.tolist()
        serialized_data.append({
            "reward_vector": reward_vector,
        })
       
    return serialized_data
class MCTSNode:
    def __init__(self, algorithm, code, obj: List[float], individual=None, depth=0, is_root=False, parent=None, visit=0, raw_info=None):
        self.algorithm = algorithm
        self.code: str = code
        self.parent: MCTSNode = parent
        self.individual = individual
        self.depth: int = depth
        self.rewards_collected: List[List[float]] = []
        self.children: List[MCTSNode] = []  # list of MCTSNode class
        self.children_info: List[dict] = []  # Raw info dictionaries of children, often used for prompting LLMs
        self.visits: int = visit
        self.subtree: List[MCTSNode] = []
        self.raw_info: List[MCTSNode] = raw_info
        self.reward_vector: List[float] = -1 * np.array(obj)  

    def add_child(self, child_node: MCTSNode):
        self.children.append(child_node)

    def __repr__(self):
        return f"MCTSNode(answer={self.algorithm}, visits={self.visits})"


class MCTS:
    def __init__(self, root_answer: Any, num_objectives: int, exploration_constant_0: float = 0.1, alpha: float = 0.5):
        self.exploration_constant_0 = exploration_constant_0  # Parameter for UCB
        self.num_objectives = num_objectives
        self.max_depth = 10
        self.epsilon = 1e-10
        self.alpha = alpha # used for progressive widenning
        self.root = MCTSNode(algorithm=root_answer, code=root_answer, obj=[0.0] * num_objectives,
                             is_root=True)
        self.global_pareto_front: List[List[float]] = []
        self.rewards = []
        self.selected_nodes: List[MCTSNode] = []
        self.rank_list = []
        self.max_bounds = [-6.28943985201484, -0.0010607615113258362]
        self.min_bounds = [-36.32015520611756, -0.46517492458224297]
        
    @staticmethod
    def dominates(reward_a: List[float], reward_b: List[float]) -> bool: 
        '''
        Args:
            Minimalization Problem, after normalization reward becomes positive
        '''
        if reward_a is None or reward_b is None:
            return False
        
        is_strictly_better_on_at_least_one = False
        for i in range(len(reward_a)):
            if reward_a[i] > reward_b[i]:  
                return False 
            if reward_a[i] < reward_b[i]:  
                is_strictly_better_on_at_least_one = True
        return is_strictly_better_on_at_least_one

    @staticmethod
    def is_non_dominated(rewards: List[List[float]], new_reward: List[float]) -> bool:
       
        for r in rewards:
            if MCTS.dominates(r, new_reward):
                return False
        return True
    
    def update_pareto_front(self, new_reward: List[float]) -> List[List[float]]:
        """
        Updates the global Pareto front with a new reward vector,
        maintaining only non-dominated solutions.
        """
        # If new_reward is dominated by the current front, return unchanged
        if not self.is_non_dominated(self.global_pareto_front, new_reward):
            print(f"Dominated solution, pareto front keep the same, pareto front is: {self.global_pareto_front}")
            return self.global_pareto_front

        # Otherwise, add new_reward and prune dominated ones
        updated_front = [r for r in self.global_pareto_front if not self.dominates(new_reward, r)]
        updated_front.append(new_reward)

        # Update the global archive in place
        self.global_pareto_front = updated_front
        
        print(f"Updated pareto front: {self.global_pareto_front}")
        return self.global_pareto_front

    def normalize(self, raw_reward_vector: list[float, float]) -> list[float, float]:
        
        normalized_scores = []
        for i, obj in enumerate(raw_reward_vector):
           # use global min and max bounds
            norm_obj = (obj - self.min_bounds[i]) / (self.max_bounds[i] - self.min_bounds[i])
            normalized_scores.append(norm_obj)
        print(f"Normalized score is: {normalized_scores}")
        return normalized_scores
            
    def backpropagate(self, node: MCTSNode, reward_vector: List[float]):
        
        current_node = node
        while current_node:
            current_node.visits += 1
            current_node.rewards_collected.append(reward_vector)
            current_node = current_node.parent

    def _calculate_hypervolume(self, front: List[List[float]]) -> float: 
        
        # Front here is already positive, resulted from the normalization process
        print(f"Front to calculate HV: {front}") # so this is negative
        
        if not front:
            return 0.0
        # We must normalize this 
        front_array = np.array(front) # positive
        
        z_ideal = np.min(front_array, axis = 0) # 2 dim
        z_nadir = np.max(front_array, axis = 0) # 2 dim
        
        print(f"Z_ideal: {z_ideal}, Z_nadir: {z_nadir}")
                
        metric = Hypervolume(ref_point= np.array([1.1, 1.1]),
                        norm_ref_point=False,
                        zero_to_one=False, # tell to normalize all points to [0, 1]
                        ideal=z_ideal,
                        nadir=z_nadir)
        
        hv = metric(front_array)
        print(f"Final HV indicator for current front: {hv}")
        return hv

    def _calculate_projection_and_penalty(self, reward_vector: List[float],
                                        pareto_front: List[List[float]],
                                        reference_point: List[float]) -> Tuple[List[float], float]: # do not pass anything here
        '''
        Args:
            High level idea: calculate the distance from a dominated solution (reward_vector) to pareto front
        '''
        for p in pareto_front:
            if np.array_equal(p, reward_vector):
                return reward_vector, 0.0

        # 2. Sort the Pareto front by the first objective
        #    This is crucial for defining the piecewise linear envelope.
        sorted_front = sorted(pareto_front, key=lambda x: x[0], reverse=True)
        
        # 3. Define the line of sight from the reference point through the reward vector
        #    Let r = reward_vector, z = reference_point. The line is r + t*(r - z)
        r = np.array(reward_vector)
        z = np.array(reference_point)
        line_dir = r - z
        
        # 4. Find the intersection of this line with the Pareto front's envelope
        print("Start performing calculate projection and penalty")
        for i in range(len(sorted_front) - 1):
            p1 = np.array(sorted_front[i])
            p2 = np.array(sorted_front[i+1])
            
            # Define the line segment between two consecutive Pareto points
            front_dir = p2 - p1
            
            # Calculate intersection using line-line intersection formula
            # This is a key geometric step
            denom = line_dir[0] * front_dir[1] - line_dir[1] * front_dir[0]
            if abs(denom) > 1e-9: # Avoid division by zero
                t = ((p1[0] - z[0]) * front_dir[1] - (p1[1] - z[1]) * front_dir[0]) / denom
                u = -((p1[0] - z[0]) * line_dir[1] - (p1[1] - z[1]) * line_dir[0]) / denom
                
                if 0 <= t and 0 <= u <= 1:
                    projection_point = p1 + u * front_dir
                    penalty = np.linalg.norm(r - projection_point)
                    print(f"penalty score is: {penalty}")
                    return projection_point.tolist(), float(penalty)
        print(f"penalty score is 0")        
        return reward_vector, 0.0 # Return default values

    def _calculate_multi_objective_ucb(self, child: MCTSNode, parent_visits: int) -> List[float]:
        
        avg_reward = []
        for i in range(self.num_objectives):
            avg = sum(r[i] for r in child.rewards_collected) / child.visits if child.visits > 0 else 0.0
            avg_reward.append(avg)
            
        print(f"Avg_reward for dim before normalization: {avg_reward}")
        normalized_avg_reward = self.normalize(avg_reward) # turn to positive
        print(f"Normalized_Avg_reward for dim before normalization: {normalized_avg_reward}")
        
        exploration_term = self.exploration_constant_0 * math.sqrt(
            math.log(parent_visits + 1) / (child.visits + self.epsilon)
        )
        print(f"Exploration term: {exploration_term}")
        ucb_vector = [normalized_obj + exploration_term for normalized_obj in normalized_avg_reward]
        
        print(f"Final UCB Vector: {ucb_vector}")
        return ucb_vector # value after adding the reward to exploration term, so the runtime can be positive sometimes
    
    
    def best_child(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Selects the best child node using the MOMCTS-hv value function.
        """
        if not node.children:
            return None

        best_child = None
        best_w_score = -float('inf')

        for child in node.children:
            if child.visits == 0:
                # Prioritize unvisited nodes for pure exploration
                return child
            
            r_sa = self._calculate_multi_objective_ucb(child, node.visits) # from a node and its parent visit, calculate the ucb vector
            # r_sa is normalized
            print(f"R_sa is: {r_sa}")
            # 2. Check for dominance against the current global Pareto front
            is_dominated = any(self.dominates(p, r_sa) for p in self.global_pareto_front)
            
            # 3. Calculate the W(s,a) score based on the MOMCTS-hv formula
            temp_front = self.update_pareto_front(r_sa)
            
            hypervolume_with_child = self._calculate_hypervolume(temp_front) # add r_sa to current pareto front 
            
            if not is_dominated:
                # Case 1: Non-dominated solution
                w_score = hypervolume_with_child
            else:
               
                _, penalty = self._calculate_projection_and_penalty(r_sa, temp_front, reference_point=[1.5] * self.num_objectives)
                w_score = hypervolume_with_child - penalty
                
            # 4. Select the child with the highest W(s,a) score
            print(f"Current w score of child: {w_score}")
            if w_score > best_w_score:
                best_w_score = w_score
                best_child = child
        print(f"Choose the best child with score: {best_w_score}")
        return best_child
    
