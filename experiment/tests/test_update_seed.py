
import sys 
sys.path.append('../automated_evidence_retrieval_study')
import pandas as pd 
from libraries.simulation_study_functions import simulation_study_functions 
from libraries.automated_handsearch import automated_handsearch
import mlflow
import logging
from libraries.review_preparation import original_review
from datetime import datetime




#read in excel workbook with data on original review
file_path = 'review_data/sysreview_dev_2SR.xlsx'
file_path_updated = 'review_data/sysreview_dev_2SR_updated.xlsx'
original_review_instance = original_review(file_path, file_path_updated)
api_choice = ['semanticscholar','openalex'] 

#loop through original systematic reviews and run automated handsearch 
for original_review_id in original_review_instance.basic_data['id']:
    print('Conducting automated citation search for original review: %s', original_review_id)
    seed_candidates = pd.DataFrame()
    for api in api_choice: 
        handsearch_cls = automated_handsearch(api)
        print('Using API: {} for {} '.format(api, original_review_id))
        seed_candidates = original_review_instance.seed_candidates.query('original_sr_id == @original_review_id').copy()

        print('Retrieving generic paper details for seed articles.')
        seed_candidates[['seed_title', 'year_published', 'seed_api_id', 'citations', 'references']] = handsearch_cls.retrieve_generic_paper_details(seed_candidates)[['title', 'year', 'paper_Id', 'referenceCount', 'citationCount']]
        original_review_instance.update_workbook_with_seed_article_data(seed_candidates, api, original_review_id)
