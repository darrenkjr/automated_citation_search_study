import asyncio
import sys 
sys.path.append('../automated_evidence_retrieval_study')
import pandas as pd
import os 
from libraries.automated_handsearch import automated_handsearch
from libraries.simulation_study_functions import simulation_study_functions 
import pandas as pd 
import time 
from datetime import timedelta 

def citation_search_test(): 
    seed_path = os.path.join(os.path.dirname(__file__),'test_data/PCOS_test_automated_handsearch_seed_8.csv')
    included_article_path = os.path.join(os.path.dirname(__file__),'test_data/PCOS_sex_func_included.csv')
    df_seed = pd.read_csv(seed_path)
    seed_sample_size = len(df_seed)
    df_included = pd.read_csv(included_article_path, dtype = object)
    api = 'openalex'
    handsearch_cls = automated_handsearch(api)
    included_paper_details = handsearch_cls.retrieve_generic_paper_details(df_included)
    
    full_results = handsearch_cls.run_citation_search(df_seed)

    #compare with included articles 
    sf = simulation_study_functions() 
    recall,precision, f1_score = sf.eval_metrics(included_paper_details,full_results)
    return full_results, recall, precision, f1_score, seed_sample_size, api

start_time = time.monotonic()
results,recall, precision, f1_score, seed_sample_size, api = citation_search_test()
end_time = time.monotonic()

duration = timedelta(seconds=end_time - start_time)
print('Recall: ', recall, ' Precision: ', precision, ' F1 score: ', f1_score, ' Seed Sample Size: ', seed_sample_size, ' API: ', api, ' Code Execution Time: {}'.format(duration) )

