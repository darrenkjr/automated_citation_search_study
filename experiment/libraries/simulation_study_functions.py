import pandas as pd
from numpy import NaN
import asyncio
import platform
import mlflow 
import logging
from itertools import combinations, chain

class simulation_study_functions: 
    
    '''class containing convenience functions for various parts of the evaluation study'''

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if platform.system()=='Windows':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                    

    def f_beta_score (self,recall, precision, beta):
        return (1+beta**2)*(recall*precision)/((beta**2)*recall+precision)


    def eval_metrics(self,original,retrieved,seed_id):

        self.logger.info('Evaluating metrics for seed ID(s)..')

        if retrieved is None:
            self.logger.warning('Retrieved dataframe is None, check API calls for the following seed_id:{}'.format(str(seed_id)))
            print('Retrieved dataframe is None, check API calls for the following seed_id:{} '.format(str(seed_id)))
            recall = NaN
            precision = NaN
            f1_score = NaN
            f2_score = NaN 
            f3_score = NaN
            f10_score = NaN
    
        
        elif retrieved.empty == False: 

            if len(set(retrieved['paper_Id']).intersection(set(original['paper_Id']))) == 0:
                self.logger.warning('No intersection found between retrieved and original articles. Setting recall, precision and f1 score to 0') 
                recall = 0 
                precision = 0 
                f1_score = 0
                f2_score = 0
                f3_score = 0
                f10_score = 0
            
            else:  
                recall = len(set(retrieved['paper_Id']).intersection(set(original['paper_Id'])))/len(original['paper_Id'])
                precision = len(set(retrieved['paper_Id']).intersection(set(original['paper_Id'])))/len(retrieved['paper_Id'])
                f1_score = self.f_beta_score(recall,precision,1)
                f2_score = self.f_beta_score(recall,precision,2)
                f3_score = self.f_beta_score(recall,precision,3)
                f10_score = self.f_beta_score(recall,precision,10)

        elif retrieved.empty == True: 
            self.logger.warning('Retrieved dataframe is empty for {}. Setting recall, precision and f1 score to NaN'.format(str(seed_id)))
            recall = NaN
            precision = NaN
            f1_score = NaN
            f2_score = NaN 
            f3_score = NaN
            f10_score = NaN

        
    
        print('Evaluation done for seed ID(s):{}'.format(str(seed_id)))
        return recall,precision,f1_score, f2_score, f3_score, f10_score
 
    def record_run(self, original_review_id, api, iter_num, seed_Id, recall, precision, f1score, f2score, f3score,f10score):

        '''
        records details of run into mlflow. Note that run details is a dictionary of the form containing details of seed candidates and the resulting run 
        ''' 
        params = {
                'original_review_id': original_review_id,
                'seed_num': len(seed_Id),
                'api': api,
                'iter_num': iter_num,
                'seed_id': seed_Id
            }

        metrics = {
                'recall': recall,
                'precision': precision,
                'f1_score': f1score,
                'f2_score': f2score,
                'f3_score': f3score,
                'f10_score': f10score
            }
        
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

    def powerset(self,iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

    def combine_citation_networks(self, citation_network_df, included_articles, original_review_id, api, iter_num): 

        '''Iteratively combines citation networks together until recall threshold is achieved'''

       
        print('Generating powerset of citation networks...')
        result_list = [] 
        #generate powerset of citation networks 
        for x in self.powerset(citation_network_df.index):
            # combined_id = [citation_network_df.loc[,'seed_Id'] , citation_network_df.loc[j,'seed_Id']]
            combined_id = citation_network_df.loc[list(x),'id'].tolist()
            combined_citation_network = pd.concat(list(citation_network_df.loc[list(x), 'citation_network_results']))
            combined_citation_network = combined_citation_network.drop_duplicates(subset=['paper_Id'])  
            #extract only the paper id column 
            combined_citation_network_result_id = combined_citation_network[['paper_Id']]
            # data = {'id': combined_id, 'citation_network_results': combined_citation_network_result_id}
            results = self.eval_metrics(included_articles,combined_citation_network_result_id,combined_id)
            # results = new_df.apply(lambda x : self.eval_metrics(included_articles,x['citation_network_results'], x['id']),axis=1)
            recall = results[0]
            precision = results[1]
            f1_score = results[2]
            f2_score = results[3]
            f3_score = results[4]
            f10_score = results[5]

            result_dict = {'original_review_id':original_review_id ,'seed_id': combined_id, 'api' : api, 'combined_citation_network_results': combined_citation_network_result_id, 'recall': recall, 'precision': precision, 'f1_score': f1_score, 'f2_score': f2_score, 'f3_score': f3_score, 'f10_score': f10_score}
            self.record_run(original_review_id, api, iter_num, combined_id, recall, precision, f1_score, f2_score, f3_score, f10_score)

            result_list.append(result_dict)
        combined_citation_network_eval_results = pd.DataFrame(result_list)
        return combined_citation_network_eval_results

        
        







    



            

            