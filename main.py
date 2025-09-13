from llm4ad.task.optimization.mo_tsp_construct import MOTSPEvaluation
from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.llm_api_gemini import GeminiApi
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.method.momcts import MOMCTS_AHD, MAProfiler
from llm4ad.method.meoh import MEoH, MEoHProfiler
from llm4ad.method.nsga2 import NSGA2, NSGA2Profiler
import os
from dotenv import load_dotenv

load_dotenv()

api_keys = []

i = 1
while True:
    key = os.getenv(f"API_KEY{i}")
    if not key:
        break
    api_keys.append(key.strip())
    i += 1


if __name__ == '__main__':
    llm = GeminiApi(
        keys=api_keys,
        model='gemini-2.5-flash',
        timeout=60
    )
    task = MOTSPEvaluation()
    method = MOMCTS_AHD(
        llm=llm,
        profiler=MAProfiler(log_dir='logs/momcts', log_style='complex'),
        evaluation=task,
        max_sample_nums=200, # max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not) or reach 'max_generations',
        max_generations=20,
        pop_size=10, # 20
        num_samplers=2,
        num_evaluators=2,
        selection_num=2, # change to 5 for meoh  
        )
    method.run()
