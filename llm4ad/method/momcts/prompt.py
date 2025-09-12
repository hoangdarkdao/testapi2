from __future__ import annotations

import copy
from typing import List, Dict

from ...base import *
from llm4ad.method.momcts.mo_mcts import MCTSNode

class MAPrompt:
    @classmethod
    def create_instruct_prompt(cls, prompt: str) -> List[Dict]:
        content = [
            {'role': 'system', 'message': cls.get_system_prompt()},
            {'role': 'user', 'message': prompt}
        ]
        return content

    @classmethod
    def get_system_prompt(cls) -> str:
        return ''

    @staticmethod
    def _format_score(score):
        """Format score to a comma-separated string of objective values.
        Handles both iterable (multi-objective) and scalar scores.
        We negate values as the original code displayed -score.
        """
        try:
            # try to iterate
            return ', '.join(str(-v) for v in score)
        except Exception:
            return str(-score)

    @classmethod
    def get_prompt_i1(cls, task_prompt: str, template_function: Function):
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content
        prompt_content = f'''{task_prompt}
1. First, describe the design idea and main steps of your algorithm in one sentence. The description must be inside within boxed {{}}. 
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e1(cls, task_prompt: str, indivs: List[Function], template_function: Function):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += (
                f'No. {i + 1} algorithm and the corresponding code are:\n'
                f'{indi.algorithm}\n{str(indi)}\n'
                f'Objective values: {cls._format_score(indi.score)}\n'
            )
        # create prmpt content
        prompt_content = f'''{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please create a new algorithm that has a totally different form from the given algorithms. Try generating codes with different structures, flows or algorithms. The new algorithm should have relatively better objective values (lower is better) across solution quality and runtime.
1. First, describe the design idea and main steps of your algorithm in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the idea in the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e2(cls, task_prompt: str, indivs: List[Function], template_function: Function):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += (
                f'No. {i + 1} algorithm and the corresponding code are:\n'
                f'{indi.algorithm}\n{str(indi)}\n'
                f'Objective values: {cls._format_score(indi.score)}\n'
            )
        # create prmpt content
        prompt_content = f'''{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please create a new algorithm that has a similar form to the No.{len(indivs)} algorithm and is inspired by the No.{1} algorithm. The new algorithm should have objective values better than both algorithms (considering solution quality and runtime trade-off).
1. Firstly, list the common ideas in the No.{1} algorithm that may give good performances.
2. Secondly, based on the common idea, describe the design idea based on the No.{len(indivs)} algorithm and main steps of your algorithm in one sentence. The description must be inside within boxed {{}}.
3. Thirdly, implement the idea in the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function):
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''

        # create prmpt content
        prompt_content = f'''{task_prompt}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please create a new algorithm that has a different form but can be a modified version of the provided algorithm. Attempt to introduce more novel mechanisms and new equations or programme segments.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the idea in the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function):
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prmpt content
        prompt_content = f'''{task_prompt}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please identify the main algorithm parameters and help me in creating a new algorithm that has different parameter settings to equations compared to the provided algorithm.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the idea in the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_s1(cls, task_prompt: str, indivs: List[Function], template_function: Function):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += (
                f"No. {i + 1} algorithm's description and the corresponding code are:\n"
                f"{indi.algorithm}\n{str(indi)}\n"
                f"Objective values: {cls._format_score(indi.score)}\n"
            )
        # create prmpt content
        prompt_content = f'''{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that is inspired by all the above algorithms with its objective values better than any of them (consider both solution quality and runtime).
1. Firstly, list some ideas in the provided algorithms that are clearly helpful to a better algorithm.
2. Secondly, based on the listed ideas, describe the design idea and main steps of your new algorithm in one sentence. The description must be inside within boxed {{}}.
3. Thirdly, implement the idea in the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content
    
    @classmethod
    def get_short_term_reflection_prompt(cls, task_prompt: str, node_set: list[MCTSNode], template_function: Function):
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        
        # Extract the 'Function' objects from the MCTSNode objects
        indivs = [node.individual for node in node_set if hasattr(node, 'individual')]

        # Create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += (
                f"No. {i + 1} algorithm's description and the corresponding code are:\n"
                f"{indi.algorithm}\n{str(indi)}\n"
                f"Objective values: {cls._format_score(indi.score)}\n"
            )
            
        # Create prompt content
        prompt_content = f'''{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that is inspired by all the above algorithms with its objective values better than any of them.
1. Firstly, list some ideas in the provided algorithms that are clearly helpful to a better algorithm.
2. Secondly, based on the listed ideas, describe the design idea and main steps of your new algorithm in one sentence. The description must be inside within boxed {{}}.
3. Thirdly, implement the idea in the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content
    
    
    @classmethod
    def get_long_term_reflection_prompt(cls, task_prompt: str, indivs: List[Function], template_function: Function):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += (
                f"No. {i + 1} algorithm's description and the corresponding code are:\n"
                f"{indi.algorithm}\n{str(indi)}\n"
                f"Objective values: {cls._format_score(indi.score)}\n"
            )
        # create prmpt content
        prompt_content = f'''{task_prompt}
Based on the best algorithms I have found so far, please help me create a new algorithm that is inspired by the successful ideas in these solutions.
The new algorithm should aim to achieve better objective values than the provided ones (consider both solution quality and runtime).
Here are the existing algorithms with their codes:
{indivs_prompt}
1. Firstly, list some of the key ideas in these provided algorithms that are clearly helpful to a better algorithm.
2. Secondly, based on the listed ideas, describe the design idea and main steps of your new algorithm in one sentence. The description must be inside within boxed {{}}.
3. Thirdly, implement the idea in the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content
    
    
    @classmethod
    def get_prompt_fix_code(cls, task_prompt: str, broken_code: str, template_function: Function, error: str | None = None):
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''  # giữ nguyên chữ ký + docstring rỗng để ép đúng signature
        err_text = f"\nKnown error (if any):\n{error}\n" if error else ""
        prompt_content = f"""{task_prompt}
The following Python function code is malformed or fails to run.{err_text}
Your job is to FIX it so it is valid Python 3 and matches exactly the template signature below. 
- Do NOT change the function name, parameters, or return contract.
- Keep logic faithful if possible; if unknown, choose a safe deterministic implementation.
- Use 4-space indentation only. No tabs. No backticks. No extra text.

Template signature to satisfy:
{str(temp_func)}

Broken code to fix:
{broken_code}

Return ONLY the corrected function code (no explanations).
"""
        return prompt_content
