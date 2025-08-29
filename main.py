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
api_keys = [
"AIzaSyAVk840U-kKUUhJymwq45k0gtzLTn3I-RQ",
"AIzaSyDF12Ka91WXxWofUOA6HwV5eq3nNTO0SV4",
"AIzaSyBtd1qP3h1lZTZoOkd39gz-6lrK5_-pKJ0",
"AIzaSyBBrAeDW_vBNfqdGyMq9f7vfuI6LejMe_k",
"AIzaSyD5cnBI_N2_mulF9s2zznmO8nW7b_vS-k0",
"AIzaSyDkt8ePFuj9oersAz6hcOF87z001-pAEYM",
"AIzaSyBbFP_0CWMmZcbjZ4GbtGdJJCKwXd0XxmM",
"AIzaSyDY091XSTuTbs1kPg5iIxGfVXNRkBxNImk",
"AIzaSyAvcJPNg0_2FNmdtbODyVyevBOMX3GieGY",
"AIzaSyAJedYZYJhuSksB8WWWEjkSgiSI3l2PLh4",
]

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