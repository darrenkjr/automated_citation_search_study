import pandas as pd 
from libraries.simulation_study_functions import simulation_study_functions 
from libraries.automated_handsearch import automated_handsearch
import mlflow
import logging 
from libraries.review_preparation import original_review
from datetime import datetime
import numpy as np
import json
from pathlib import Path


class main: 

    def __init__(self): 
        pass 

    def run_experiment(self): 
        #todays date and time in str format 
        date_time = datetime.now().strftime("%d_%m_%Y_%H_%M")
        current_batch = 'cee'
        data_dir = Path("review_data")
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        #read in excel workbook with data on original review
        api_choice = ['openalex']

        file_path = data_dir / f"sr_samples_full_{current_batch}.xlsx"
    
        #setting up experiment in mlflow
        experiment_name = date_time + '_' + 'citation_search_' + current_batch + '_full' + api_choice[0]
        current_experiment = mlflow.set_experiment(experiment_name)


        #set up logging 
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename = r'logs/' + experiment_name,
                            level = logging.WARNING,
                            format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')


        print('Instantiating original review class')
        
        original_review_instance = original_review(file_path, current_batch, api_choice)
        #preps data re: original review, calculating recall, and retrieving data for included articles from openalex api

        original_review_instance.prep_data()
        experiment_tags = {"version": current_batch, 
                        "number of ref reviews" : len(original_review_instance.basic_data['id'].unique()),
                        "seed_candidates" : 'full',  
        }
        mlflow.set_experiment_tags(experiment_tags)

        #loop through original systematic reviews and run automated handsearch 
        for original_review_id in original_review_instance.basic_data['id']:
            print('Conducting automated citation search for original review: %s', original_review_id)
            self.logger.info('Conducting automated citation search for original review: %s', original_review_id)
            #get year limits for current original review 
            year_limit_start = original_review_instance.basic_data.query('id == @original_review_id')['selection_criteria_year_limit_start'].values[0]
            year_limit_end = original_review_instance.basic_data.query('id == @original_review_id')['selection_criteria_year_limit_end'].values[0]
            # result_path = 'results/combined_citation_network_results/{}_{}.json'.format(experiment_name,(original_review_id).replace(['/','.'], '_'))
            #check if year_limit_start is NaN 
            if np.isnan(year_limit_start):
                year_limit_start = 0
            for api in api_choice: 
                mlflow.set_experiment_tags({'api' : api})
                #get seed candidates and included articles for current original review    
                seed_candidates = original_review_instance.seed_candidates.query('original_sr_id == @original_review_id').copy()
                included_articles = original_review_instance.included_articles.query('original_sr_id == @original_review_id')
                # remove duplicate between seed and included articles from seed_articles 
                seed_candidates = seed_candidates[~seed_candidates['seed_Id'].isin(included_articles['included_doi'].tolist())]
                
                iter_num = 1
                handsearch_cls = automated_handsearch(api)
                print('Using API: {} for {} '.format(api, original_review_id))

                if api == 'openalex':
                    included_article_data = included_articles 
                elif api == 'semanticscholar' : 
                    self.logger.info('Retrieving generic paper details for included articles as API is semantic scholar')
                    print('Retrieving generic paper details for included articles as API is semantic scholar.')
                    included_article_data = handsearch_cls.retrieve_generic_paper_details(included_articles)

                print('Retrieving generic paper details for seed articles.')
                self.logger.info('Retrieving generic paper details for seed articles.')

                # #filter seed candidates to only include those with citation count below 20000
                seed_candidates = seed_candidates.query('citation_limit_exceeded == 0').reset_index(drop = True)
                seed_candidates = seed_candidates.query('no_data_from_api == 0').reset_index(drop = True)
    
                print('Step 1 : Run Citation Search')
                print('Retrieving Citations for {}'.format(original_review_id))
                self.logger.info(str('Generating citation networks using:{}'.format(api)))
                citation_network_results = handsearch_cls.run_citation_search(seed_candidates,year_limit_start,year_limit_end)
                seed_candidates['citation_network_results'] = citation_network_results 

                print('Step 2 : Evaluate results (recall, precision and f1-3 score)')
                # results = seed_candidates.apply(lambda x : sf.eval_metrics(included_article_data,x['citation_network_results']), axis = 1, result_type = 'expand')

                sf = simulation_study_functions()

                if 'seed_Id' in seed_candidates.columns:
                    seed_candidates.rename(columns = {'seed_Id' : 'id'}, inplace = True)
                    seed_candidates['id'].fillna('pmid:'+ seed_candidates['seed_pmid'], inplace = True)
                results = seed_candidates.apply(lambda x : pd.Series(sf.eval_metrics(included_article_data,x['citation_network_results'], x['id'])), axis = 1)
                seed_candidates['recall'] = results[results.columns[0]]
                seed_candidates['precision'] = results[results.columns[1]]
                seed_candidates['f1_score'] = results[results.columns[2]]
                seed_candidates['f2_score'] = results[results.columns[3]]
                seed_candidates['f3_score'] = results[results.columns[4]]
                seed_candidates['f10_score'] = results[results.columns[5]]
                print('Step 3: Record results of citation search for each seed article with mlflow') 
                seed_candidates.apply(lambda x : sf.record_run(original_review_id, api, iter_num, x['id'], x['recall'],x['precision'], x['f1_score'], x['f2_score'], x['f3_score'], x['f10_score']), axis = 1)

                print('Step 4: Check results (recall) and identify seed articles that have zero recall')
                seed_candidates_filtered = seed_candidates[seed_candidates['recall'] > 0]
    
                #Step 5 : Sorting by recall, and then choosing top 10 if more than 10 articles have non zero recall 
                print('Step 5: Sorting by recall and selecting top 10')
                if len(seed_candidates_filtered) > 10:
                    seed_candidates_filtered = (seed_candidates_filtered.sort_values(by = 'recall', ascending = False).head(10))
                else: 
                    seed_candidates_filtered = seed_candidates_filtered.sort_values(by = 'recall', ascending = False)
            
                #Step 6: Combine different citation networks for each non zero recall seed article until optimum recall is reached
                print('Step 6: Combining citation networks..')
                combined_results = sf.combine_citation_networks(seed_candidates_filtered,included_article_data, original_review_id, api, iter_num)
                print('Citation search completed for original review: %s', original_review_id)
                print('exportin results as json')
                #export results as json to result_path 
                # combined_results.to_json(result_path)


            print('Done - Conducting experimental runs on next api')
        print('Done - Executing next original review')
        
                
        #export all recorded runs in mlflow experiment to csv 
            

#run experiment 
if __name__ == "__main__":
    main_instance = main()
    main_instance.run_experiment()
    mlflow.set_experiment_tags({'completed' : 'yes'})
    print('Completed')
