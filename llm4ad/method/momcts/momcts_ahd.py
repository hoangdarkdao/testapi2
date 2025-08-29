from __future__ import annotations
import json
import logging
import os
import numpy as np
import concurrent.futures
import copy
import time
import random
import time
import traceback
from threading import Thread
from typing import Optional, Literal
from .extension import dominates, hypervolume_contribution
from .population import Population
from .mo_mcts import MCTS, MCTSNode
from .profiler import MAProfiler
from .prompt import MAPrompt
from .sampler import MASampler
from ...base import (
    Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator
)
from ...tools.profiler import ProfilerBase


# momcts_ahd.py

# ... (các import và __init__ như cũ)

class MOMCTS_AHD:
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: ProfilerBase = None,
                 max_sample_nums: Optional[int] = 100,
                 init_size: Optional[float] = 4,
                 pop_size: Optional[int] = 10,
                 selection_num: int = 2,
                 num_samplers: int = 1,  # the number of threads to sample in parallel
                 num_evaluators: int = 1,
                 alpha: float = 0.5,
                 lambda_0: float = 0.1,
                 *,
                 resume_mode: bool = False,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: Literal['thread',
                                                       'process'] = 'thread',
                 **kwargs):
        """Evolutionary of Heuristics.
        Args:
            llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
            evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
            profiler        : an instance of 'llm4ad.method.eoh.EoHProfiler'. If you do not want to use it, you can pass a 'None'.
                              pass 'None' to disable this termination condition.
            max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not) or reach 'max_generations',
                              pass 'None' to disable this termination condition.
            init_size       : population size, if set to 'None', EoH will automatically adjust this parameter.
            pop_size        : population size, if set to 'None', EoH will automatically adjust this parameter.
            selection_num   : number of selected individuals while crossover.
            alpha           : a parameter for the UCT formula, which is used to balance exploration and exploitation.
            lambda_0        : a parameter for the UCT formula, which is used to balance exploration and exploitation.
            resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
            debug_mode      : if set to True, we will print detailed information.
            multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
                multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
                setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
                Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
                and you set this argument to 'thread'.
            **kwargs                    : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.
        """
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self.lambda_0 = lambda_0
        self.alpha = alpha
        self._init_pop_size = init_size
        self._pop_size = pop_size
        self._selection_num = selection_num
        self.output_path = "logs/momcts/archive"
        # samplers and evaluators
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._debug_mode = debug_mode
        llm.debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval

        # function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(
            self._template_program_str)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(
            self._template_program_str)

        # adjust population size
        self._adjust_pop_size()

        # population, sampler, and evaluator
        self._population = Population(
            init_pop_size=init_size, pop_size=self._pop_size)
        self._sampler = MASampler(llm, self._template_program_str)
        self._evaluator = SecureEvaluator(
            evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler

        # statistics
        self._tot_sample_nums = 0  # the current function that is evaluated

        # reset _initial_sample_nums_max
        self._initial_sample_nums_max = min(
            self._max_sample_nums,
            10 * self._init_pop_size
        )

        # multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )

        # pass parameters to profiler
        if profiler is not None:
            self._profiler.record_parameters(
                llm, evaluation, self)  # ZL: necessary
    
    def _perform_short_term_reflection(self, node: MCTSNode):
        """
        Thực hiện short-term reflection bằng cách sử dụng thông tin từ
        nút hiện tại, nút cha, và các nút anh em.
        """
        if node is None or node.parent is None or node.parent == self.mcts.root:
            return

        reflection_set = []

        # Thêm nút cha vào reflection set
        parent_individual = node.parent.individual
        if parent_individual is not None:
            reflection_set.append(parent_individual)

        # Thêm các nút anh em vào reflection set
        for sibling_node in node.parent.children:
            if sibling_node.individual is not None:
                reflection_set.append(sibling_node.individual)
        
        # Loại bỏ các cá thể trùng lặp
        unique_reflection_set = []
        unique_algorithms = []
        for individual in reflection_set:
            if str(individual) not in unique_algorithms:
                unique_reflection_set.append(individual)
                unique_algorithms.append(str(individual))
        
        if len(unique_reflection_set) < 2:
            return

        try:
            # Tạo prompt reflection từ tập hợp các cá thể đã chọn
            prompt = MAPrompt.get_short_term_reflection_prompt(
                self._task_description_str,
                unique_reflection_set,
                self._function_to_evolve
            )
            print("Performing short-term reflection...")
            self._sample_evaluate_register(prompt)
        except Exception as e:
            print(f"Error during short-term reflection: {e}")
            if self._debug_mode:
                traceback.print_exc()
    
    def _perform_long_term_reflection(self):
        """
        Thực hiện long-term reflection bằng cách tổng hợp ý tưởng từ các cá thể tốt nhất
        trong quần thể Pareto front.
        """
        # Lấy các cá thể tốt nhất
        best_individuals = self.population_management_s1(self._population.population, size=self._pop_size)
        
        # Cần ít nhất 2 cá thể để tổng hợp và so sánh có tốt không
        if len(best_individuals) < 2:
            return

        try:
            # Tạo prompt reflection từ tập hợp các cá thể tốt nhất 
            prompt = MAPrompt.get_long_term_reflection_prompt(
                self._task_description_str,
                best_individuals,
                self._function_to_evolve
            )
            print("Performing long-term reflection...")
            self._sample_evaluate_register(prompt)
        except Exception as e:
            print(f"Error during long-term reflection: {e}")
            if self._debug_mode:
                traceback.print_exc()
                
    def _adjust_pop_size(self):
        # adjust population size
        if self._max_sample_nums >= 10000:
            if self._pop_size is None:
                self._pop_size = 40
            elif abs(self._pop_size - 40) > 20:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 40.')
        elif self._max_sample_nums >= 1000:
            if self._pop_size is None:
                self._pop_size = 20
            elif abs(self._pop_size - 20) > 10:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 20.')
        elif self._max_sample_nums >= 200:
            if self._pop_size is None:
                self._pop_size = 10
            elif abs(self._pop_size - 10) > 5:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 10.')
        else:
            if self._pop_size is None:
                self._pop_size = 5
            elif abs(self._pop_size - 5) > 5:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 5.')

    def _sample_evaluate_register(self, prompt, func_only=False):
       
        sample_start = time.time()
        thought, func = self._sampler.get_thought_and_function(
            self._task_description_str, prompt)
        sample_time = time.time() - sample_start
        if thought is None or func is None:
            print("New return in _sample_evaluate_register, momcts_ahd.py")
            return False
        # convert to Program instance
        program = TextFunctionProgramConverter.function_to_program(
            func, self._template_program)
        if program is None:
            return False
        # evaluate
        score, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time,
            program
        ).result()  # this is where we update function score #
        # register to profiler
        # evaluate_program_record_time
        func.score = score
        func.evaluate_time = eval_time
        func.algorithm = thought
        func.sample_time = sample_time
        if self._profiler is not None:
            self._profiler.register_function(
                func, program=str(program))  # just write log to json
            if isinstance(self._profiler, MAProfiler):
                self._profiler.register_population(
                    self._population)  # just write log to json
            self._tot_sample_nums += 1  # the current function that is evaluated
        if func_only:
            print("New return in _sample_evaluate_register, momcts_ahd.py")
            return func
        if func.score is None:
            print("New return in _sample_evaluate_register, momcts_ahd.py")
            return False
        # good until these
        self._population.register_function(func)  # where the population update
        # self.mcts.update_global_pareto_front(func.score, func)
        return True

    def _continue_loop(self) -> bool:
        if self._max_sample_nums is None:
            return True
        else:
            # if the current evaluation func number < max number, still do this
            return self._tot_sample_nums < self._max_sample_nums

    def check_duplicate(self, population, code: str):
        '''
        Args:
            population: is a list of function ,find in code.py
            code: is the str code like def select_next_node(...):....
        '''
        for ind in population:
            if code == ind.code:
                return True
        return False

    def check_duplicate_obj(self, population, score: list[float, float]):
        for ind in population:
            if np.array_equal(score, ind.individual.score):
                return True
        return False

    # add tree path for reasoning s1
    def population_management_s1(self, pop_input, size):
        pop = [individual for individual in pop_input if individual.score is not None]
        if size > len(pop):
            size = len(pop)
        unique_pop = []
        unique_algorithms = []
        for individual in pop:
            if str(individual) not in unique_algorithms:
                unique_pop.append(individual)
                unique_algorithms.append(str(individual))

        pareto_front = []
        for candidate in unique_pop:
            is_dominated = False
            for existing in pareto_front:
                if dominates(existing.score, candidate.score):
                    is_dominated = True
                    break
            if not is_dominated:
                # Remove dominated individuals from Pareto front
                pareto_front = [
                    existing
                    for existing in pareto_front
                    if not dominates(candidate.score, existing.score)
                ]
                pareto_front.append(candidate)

        # If Pareto front is smaller than size, fill with remaining population
        if len(pareto_front) < size:
            remaining_pop = [
                ind for ind in unique_pop if ind not in pareto_front]
            # Sort remaining population by hypervolume contribution (example - can be changed)
            remaining_pop = sorted(remaining_pop,
                                   key=lambda x: hypervolume_contribution(
                                       x.score, pareto_front),
                                   reverse=True)
            pareto_front.extend(remaining_pop[:size - len(pareto_front)])

        pop_new = pareto_front[:size]

        return pop_new

    def expand(self, mcts: MCTS, node_set: list[MCTSNode], cur_node: MCTSNode, option: str):
        
        print("Inside expand, momcts_ahd.py")
        print(f"Current node set depth: {[node.depth for node in node_set]}")
        print(f"Current node depth: {cur_node.depth}")  
        
        is_valid_func = True
        if option == 's1':
            path_set = []
            now = copy.deepcopy(cur_node)
            while now.algorithm != "Root":
                path_set.append(now.individual)
                now = copy.deepcopy(now.parent)
            path_set = self.population_management_s1(path_set, len(path_set))
            if len(path_set) == 1:
                return node_set

            i = 0
            while i < 3:
                prompt = MAPrompt.get_prompt_s1(
                    self._task_description_str, path_set, self._function_to_evolve)
                func = self._sample_evaluate_register(prompt, func_only=True)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(
                    node_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    break

        elif option == 'e1':
            indivs = [copy.deepcopy(children.subtree[random.choices(range(len(children.subtree)), k=1)[0]].individual)
                      for
                      # so mcts.root.children becauses we only use e1 in initialization
                      children in mcts.root.children]
            prompt = MAPrompt.get_prompt_e1(
                self._task_description_str, indivs, self._function_to_evolve)
            func = self._sample_evaluate_register(prompt, func_only=True)
            if func is False:
                is_valid_func = False
            else:
                is_valid_func = (func.score is not None)

        elif option == 'e2':
            i = 0
            while i < 3:
                now_indiv = None
                while True:
                    now_indiv = self._population.selection()
                    if now_indiv != cur_node.individual:
                        break
                prompt = MAPrompt.get_prompt_e2(self._task_description_str, [now_indiv, cur_node.individual],
                                                self._function_to_evolve)
                func = self._sample_evaluate_register(prompt, func_only=True)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(
                    node_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    break

        elif option == 'm1':
            i = 0
            while i < 3:
                prompt = MAPrompt.get_prompt_m1(self._task_description_str, cur_node.individual,
                                                self._function_to_evolve)
                func = self._sample_evaluate_register(prompt, func_only=True)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(
                    node_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    break

        elif option == 'm2':
            i = 0
            while i < 3:
                prompt = MAPrompt.get_prompt_m2(self._task_description_str, cur_node.individual,
                                                self._function_to_evolve)
                func = self._sample_evaluate_register(prompt, func_only=True)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(
                    node_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    break

        else:
            assert False, 'Invalid option!'

        if not is_valid_func:
            print(f"Timeout emerge, no expanding with action {option}.")
            return node_set

        if option != 'e1':
            print(
                f"Action: {option}, Father Obj: {cur_node.raw_info.score}, Now Obj: {func.score}, Depth: {cur_node.depth + 1}")
        else:
            if self.check_duplicate_obj(node_set, func.score):
                print(
                    f"Duplicated e1, no action, Father is Root, Abandon Obj: {func.score}")
            else:
                print(
                    f"Action: {option}, Father is Root, Now Obj: {func.score}")

        if is_valid_func and np.any(func.score != float('-inf')):
            self._population.register_function(func)
            print(f"Passed score into MCTSNode in expand: {func.score}")
            now_node = MCTSNode(func.algorithm, str(func), -1 * func.score, individual=func,
                                parent=cur_node, depth=1, visit=1, raw_info=func)
            if option == 'e1':
                now_node.subtree.append(now_node)
            cur_node.add_child(now_node)
            mcts.backpropagate(now_node, now_node.reward_vector)
            node_set.append(now_node)
        return node_set

    # where to update the self._population, adding it to a list
    def _iteratively_init_population_root(self):
        """Let a thread repeat {sample -> evaluate -> register to population}
        to initialize a population.
        """
        print("Inside _iteratively_init_population_root, population.py")
        while len(self._population.population) < self._init_pop_size:
            print(f"Length of current pop: {len(self._population.population)}")
            try:
                # get a new func using e1
                prompt = MAPrompt.get_prompt_e1(self._task_description_str, self._population.population,
                                                self._function_to_evolve)
                print("Initialization. Perfrom e1 operation")
                self._sample_evaluate_register(prompt)
                # The method selects and updates the elitist population — the best individuals based on non-dominated sorting. These elites are preserved across generations to maintain and improve solution quality over time.
                self._population.survival()

                if self._tot_sample_nums >= self._initial_sample_nums_max:
                    # print(f'Warning: Initialization not accomplished in {self._initial_sample_nums_max} samples !!!')
                    print(
                        f'Note: During initialization, EoH gets {len(self._population) + len(self._population._next_gen_pop)} algorithms '
                        f'after {self._initial_sample_nums_max} trails.')
                    break
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue
        print(f"Initial population length: {len(self._population.population)}")

    def _init_one_solution(self):
        print("Start init population, momcts_ahd.py")
        while len(self._population._next_gen_pop) == 0:
            print("Inside while of init population, momcts_ahd.py")
            try:
                # get a new func using i1
                prompt = MAPrompt.get_prompt_i1(
                    self._task_description_str, self._function_to_evolve)
                self._sample_evaluate_register(prompt)

            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

    def _multi_threaded_sampling(self, fn: callable, *args, **kwargs):
        """Execute `fn` using multithreading.
        In MCTS_MA, `fn` can be `self._iteratively_use_eoh_operator`.
        """
        # threads for sampling
        sampler_threads = [
            Thread(target=fn, args=args, kwargs=kwargs)
            for _ in range(self._num_samplers)
        ]
        for t in sampler_threads:
            t.start()
        for t in sampler_threads:
            t.join()

    def run(self):
        # 1. first generate one solution as initialization
        self._init_one_solution()

        # The method selects and updates the elitist population — the best individuals based on non-dominated sorting. These elites are preserved across generations to maintain and improve solution quality over time.
        self._population.survival()
        self.mcts = MCTS('Root', num_objectives=2, alpha=self.alpha,
                         exploration_constant_0=self.lambda_0)
        # 2. expand root
        # get until the current population size = initial population size
        self._iteratively_init_population_root()

        # 3. update mcts
        for indiv in self._population.population:
            print(f"Individual score: {indiv.score}")
            logging.info(f"Individual score: {indiv.score}")

            print(f"Passed score into MCTS Node: {indiv.score}") # negative
            now_node = MCTSNode(indiv.algorithm, str(indiv), -1 * indiv.score, individual=indiv,
                                parent=self.mcts.root,
                                depth=1, visit=1, raw_info=indiv)
            self.mcts.root.add_child(now_node)
            self.mcts.backpropagate(now_node, now_node.reward_vector)
            now_node.subtree.append(now_node)
            # self.mcts.update_global_pareto_front(indiv.score, now_node) # newly added
        # terminate searching if
        if len(self._population) < self._selection_num:
            print(
                f'The search is terminated since MCTS_AHD unable to obtain {self._selection_num} feasible algorithms during initialization. '
                f'Please increase the `initial_sample_nums_max` argument (currently {self._initial_sample_nums_max}). '
                f'Please also check your evaluation implementation and LLM implementation.')
            return

        # evolutionary search
        n_op = ['e1', 'e2', 'm1', 'm2', 's1']
        op_weights = [0, 1, 2, 2, 1]
        while self._continue_loop():  # if current evaluation < max evaluation, still evaluate function
            node_set = []
            # print(f"Current performances of MCTS nodes: {self.mcts.rank_list}")
            print(
                f"Current number of MCTS nodes in the subtree of each child of the root: {[len(node.subtree) for node in self.mcts.root.children]}")
            cur_node = self.mcts.root
            while len(cur_node.children) > 0 and cur_node.depth < self.mcts.max_depth:
                # change this to choose the single best child from a parent node
                next_node = self.mcts.best_child(cur_node)
                if next_node is None:
                    break

                # here is progresive something
                if int((cur_node.visits) ** self.mcts.alpha) > len(cur_node.children):
                    if cur_node == self.mcts.root:
                        op = 'e1'
                        print("Perfrom e1 operation")
                        self.expand(
                            self.mcts, self.mcts.root.children, cur_node, op)
                    else:
                        op = 'e2'
                        print("Perfrom e2 operation")
                        self.expand(self.mcts, cur_node.children, cur_node, op)

                cur_node = next_node
            
            # short-term reflection
            self._perform_short_term_reflection(cur_node)
            
            # long-term reflection
            if self._tot_sample_nums % 10 == 0 and self._tot_sample_nums > 0:
                self._perform_long_term_reflection()


            for i in range(len(n_op)):
                op = n_op[i]  # get operation
                print(f"Get operation {op}")
                print(
                    f"Iter: {self._tot_sample_nums}/{self._max_sample_nums} OP: {op}", end="|")
                # get the max number of performing that operation
                op_w = op_weights[i]
                for j in range(op_w):
                    node_set = self.expand(self.mcts, node_set, cur_node, op)
                    
            self._population.survival()
            filename = os.path.join(self.output_path, f"population_generation_{int(time.time())}.json")
    
            # Prepare the population data for JSON serialization
            population_data = []
            for indiv in self._population.population:
                f_score = indiv.score
                population_data.append({
                    "algorithm": indiv.algorithm,
                    "function": str(indiv),
                    "score": f_score.tolist()
                })

            log_directory = os.path.dirname(filename) # Extract the directory path from the full filename
            if not os.path.exists(log_directory):
                os.makedirs(log_directory, exist_ok=True) # Create all intermediate directories if they don't exist

            # Write the data to the JSON file
            with open(filename, 'w') as f:
                json.dump(population_data, f, indent=5)
        if self._profiler is not None:
            self._profiler.finish()

        self._sampler.llm.close()