
import sys 
sys.path.append('../automated_evidence_retrieval_study')
import pandas as pd 
from libraries.simulation_study_functions import simulation_study_functions
from libraries.automated_handsearch import automated_handsearch
import os 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
import numpy as np
from numpy import NaN
import logging 

class original_review:

    '''class for original review included as part of testing of automated handsearch. Instantiating the class will
    create a dataframe with IDs / DOIs of potential seed articles that have been extracted from the background section of the review,
    IDs/DOIs of the final included articles that were included in the original review, and details on search strategy
    sufficient to compute recall, precision, and f1 score
    ''' 

    def __init__(self,file_path,current_batch, api_choice):

        self.logger = logging.getLogger(__name__)
        self.file_path = os.path.join(file_path)
        self.api_choice = api_choice
        self.citation_limit = 10000
        self.current_batch = current_batch
        #read in excel workbook with data on original review 
        if self.file_path.endswith('.xlsx'):
            self.workbook_dict = pd.read_excel(self.file_path,sheet_name=None, dtype = {'included_pmid': str, 'seed_pmid':str, 'included_mag_id' : str})
            self.basic_data_with_exclude = self.workbook_dict['sys_rev_data']
            #convert all ids to lowercase
            self.basic_data_with_exclude['id'] = self.basic_data_with_exclude['id'].str.lower()
            self.included_articles = self.workbook_dict['sys_rev_included_data']

            self.included_articles.index = pd.Int64Index(self.included_articles.index)

            #convert mag id in included articles to str
            self.included_articles['included_mag_id'] = self.included_articles['included_mag_id'].astype(str) 
            #convert original_sr_id and included_doi to lowercase 
            self.included_articles['original_sr_id'] = self.included_articles['original_sr_id'].str.lower()
            self.included_articles['included_doi'] = self.included_articles['included_doi'].str.lower()
            self.seed_candidates = self.workbook_dict['sys_rev_seed_candidates']
            #remove whitespace from both all columns in both dataframes
            self.included_articles = self.included_articles.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            self.seed_candidates = self.seed_candidates.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            self.seed_candidates = self.seed_candidates.apply(lambda x : x.str.lower() if x.dtype == "object" else x)

    def prep_data(self): 
        #drow rows where exclude == 1
        self.basic_data = self.basic_data_with_exclude[self.basic_data_with_exclude['exclude'] != 1].reset_index(drop=True).rename(columns={'index': 'original_sr_id'})
        #loop through unique original ids, and calculate recall, precision and f1 score for each id
        for id in self.basic_data['id']:
            self.basic_data.loc[self.basic_data['id'] == id, 'recall'] = len(self.included_articles.query('original_sr_id == @id')) / len(self.included_articles.query('original_sr_id == @id'))
            self.basic_data.loc[self.basic_data['id'] == id, 'precision'] = len(self.included_articles.query('original_sr_id == @id')) / self.basic_data.loc[self.basic_data['id'] == id, 'original_search_retrieved']
            self.basic_data.loc[self.basic_data['id'] == id, 'f1_score'] = 2 * ((self.basic_data.loc[self.basic_data['id'] == id, 'precision'] * self.basic_data.loc[self.basic_data['id'] == id, 'recall']) / (self.basic_data.loc[self.basic_data['id'] == id, 'precision'] + self.basic_data.loc[self.basic_data['id'] == id, 'recall']))
            self.basic_data.loc[self.basic_data['id'] == id, 'f2_score'] = 5 * ((self.basic_data.loc[self.basic_data['id'] == id, 'precision'] * self.basic_data.loc[self.basic_data['id'] == id, 'recall']) / (4 * self.basic_data.loc[self.basic_data['id'] == id, 'precision'] + self.basic_data.loc[self.basic_data['id'] == id, 'recall']))
            self.basic_data.loc[self.basic_data['id'] == id, 'f3_score'] = 10 * ((self.basic_data.loc[self.basic_data['id'] == id, 'precision'] * self.basic_data.loc[self.basic_data['id'] == id, 'recall']) / (9 * self.basic_data.loc[self.basic_data['id'] == id, 'precision'] + self.basic_data.loc[self.basic_data['id'] == id, 'recall']))
            self.basic_data.loc[self.basic_data['id'] == id, 'f10_score'] = (10**2 + 1) * ((self.basic_data.loc[self.basic_data['id'] == id, 'precision'] * self.basic_data.loc[self.basic_data['id'] == id, 'recall']) / (10**2 * self.basic_data.loc[self.basic_data['id'] == id, 'precision'] + self.basic_data.loc[self.basic_data['id'] == id, 'recall']))


        #retrieve systematic review details: title, year_published, source_database from api call
        for api in self.api_choice:
            self.current_api = api 
            self.handsearch_cls = automated_handsearch(self.current_api)
            print('Retrieving generic paper details for original reviews.')

            if self.current_api == 'openalex':
                original_review_data_oa = self.handsearch_cls.retrieve_generic_paper_details(self.basic_data)
                self.basic_data['title'] = original_review_data_oa['title']
                self.basic_data['year_published'] = original_review_data_oa['year']
                self.basic_data['pmid'] = original_review_data_oa['pmid']
                self.basic_data['mag_id'] = original_review_data_oa['mag_id']
            #dumb workaround 
            elif self.current_api == 'semanticscholar':
                #change id to seed_id because seed_id has been used due to legacy reasons, and this an example of spaghetti code that i will need to clean up before paper submission
                original_review_data_ss = self.handsearch_cls.retrieve_generic_paper_details(self.basic_data)
                print('Original review data from semantic scholar: {}'.format(original_review_data_ss))
                self.basic_data['title'] = original_review_data_ss['title']
                self.basic_data['year_published'] = original_review_data_ss['year']
                self.basic_data['pmid'] = original_review_data_ss['externalIds.PubMed']
                self.basic_data['mag_id'] = original_review_data_ss['externalIds.MAG']


            #retrieves generic included article data and generate intra   
            self.intra_cluster_sim_helper()
            self.prep_seed_articles()  
            
            self.logger.info('Included article and seed article retrieval complete')
            file_path_updated = 'results/sr_samples_{}_{}.xlsx'.format(self.current_batch, self.current_api)

            #copy excel workbook and update with data on original reviews and included articles 
            with pd.ExcelWriter(file_path_updated, engine="xlsxwriter", mode="w") as writer:
                os.path.join(file_path_updated)
                if self.current_api == 'openalex': 
                    api_choice = 'oa'
                elif self.current_api == 'semanticscholar': 
                    api_choice = 'ss'
                self.basic_data.to_excel(writer, sheet_name='sys_rev_data_{}'.format(api_choice), index=False)    
                self.included_articles.to_excel(writer, sheet_name='sys_rev_included_data_{}'.format(api_choice), index=False)
                self.seed_candidates.to_excel(writer, sheet_name='sys_rev_seed_candidates_{}'.format(api_choice), index=False)
            
            self.logger.info('Excel workbook updated with data on original reviews and included articles')
    
    def drop_duplicates_keep_most_data(self, df, important_cols, duplicate_col):
        df.sort_values(by=duplicate_col + important_cols, ascending=[True] + [False], inplace=True)
        # Drop duplicates and keep the first (i.e., the one with the largest sum)
        df.drop_duplicates(subset=duplicate_col, keep='first', inplace=True)
        # Drop the 'important_sum' column as it's no longer needed
        return df
   
    def prep_seed_articles(self): 
        '''function to retrieve generic paper details for seed articles and update seed article dataframe with this data'''
        col_keep_oa = ['original_sr_id', 'id', 'seed_doi_alternate', 'seed_pmid', 'article_type', 'article_subtype', 'title', 'year',
                                                                        'cited_section', 'paper_Id', 'language', 'references', 'citations', 'type', 'api_path', 'pmid', 'mag_id', 'no_data_from_api','cited_by_api_url', 'citation_limit_exceeded', 'referenced_works']
        col_keep_ss = ['original_sr_id', 'id', 'seed_doi_alternate', 'seed_pmid', 'article_type', 'article_subtype', 'title', 'abstract', 'year',
                                                                        'cited_section', 'paper_Id', 'referenceCount', 'citationCount', 'publicationTypes', 'originating_api_path', 'externalIds.PubMed', 'externalIds.MAG', 'no_data_from_api']

        
        for original_review_id in self.basic_data['id']:
            print('Retrieving paper details for seed articles, for original review: {}'.format(original_review_id))
            seed_candidates = self.seed_candidates.query('original_sr_id == @original_review_id')
            if self.current_api == 'openalex':
                seed_candidate_generic_df_oa = self.handsearch_cls.retrieve_generic_paper_details(seed_candidates)
                seed_candidate_generic_df_oa['citation_network_size'] = seed_candidate_generic_df_oa['references'] + seed_candidate_generic_df_oa['citations']
                #if api results give duplicate rows (ie: ambiguous doi - keep the entry with the largest citation network - ie: most data, else keep the first result)
                seed_candidate_generic_df_oa = self.drop_duplicates_keep_most_data(seed_candidate_generic_df_oa, ['citation_network_size'],['sorting_ids'])
                seed_candidate_merge = pd.merge(seed_candidates, seed_candidate_generic_df_oa, left_on = 'id', right_on = 'sorting_ids', how = 'inner', suffixes = ('_input', '_oaresp'))

                #handle merging conflicts 
                rename_dct = {}
                drop_col = []
                for col in seed_candidate_merge.columns: 
                    if col.endswith('_oaresp'):
                        api_col_name = col.replace('_oaresp','')
                        rename_dct[col] = api_col_name
                        drop_col.append(api_col_name + '_input')
                
                seed_candidate_merge.rename(columns = rename_dct, inplace = True)
                seed_candidate_merge.drop(columns = drop_col, inplace = True)

                #only keep certain columns 
                
                seed_candidate_merge['citation_limit_exceeded'] = seed_candidate_merge['citations'].apply(lambda x : 1 if x > self.citation_limit else 0)
              
                if 'no_data_from_api' not in seed_candidate_merge.columns:
                    seed_candidate_merge['no_data_from_api'] = 0
                    
                seed_candidate_merge['no_data_from_api'].fillna(0, inplace = True)
                seed_candidate_merge = seed_candidate_merge[col_keep_oa].copy()
                col_rename_dct = {'paper_Id' : 'api_id', 'id' : 'seed_Id', 'title' : 'seed_title', 'year' : 'year_published'}
                seed_candidate_merge.rename(columns = col_rename_dct, inplace = True)
                #rename columns 
                
                seed_candidate_merge_oa = seed_candidate_merge.copy()
                original_review_id_index = self.seed_candidates.index[self.seed_candidates['original_sr_id'] == original_review_id]

                col_update = [col_rename_dct.get(item, item) for item in col_keep_oa]
                seed_candidate_merge_oa.index = original_review_id_index
                print('Updating seed article dataframe with generic paper details from api for original review: {}'.format(original_review_id))

                # if columns are different then merge and handle conflicts

                if set(seed_candidate_merge_oa) == set(self.seed_candidates.columns):
                    print('Columns are matching - updating dataframe')
                    self.seed_candidates.update(seed_candidate_merge_oa)
               
                elif set(seed_candidate_merge_oa.columns) != set(self.seed_candidates.columns):
                    #update
                    new_col = [col for col in seed_candidate_merge_oa.columns.to_list() if col not in self.seed_candidates.columns.to_list()]
                    print('New columns found: {}'.format(new_col))
                    #add new columns to seed article dataframe
                    for col in new_col:
                        self.seed_candidates[col] = None

                    self.seed_candidates.update(seed_candidate_merge_oa)
            
            elif self.current_api == 'semanticscholar':
                seed_candidates_copy = seed_candidates.copy()
                seed_candidates_copy['seed_Id'] = seed_candidates_copy['seed_Id'].fillna(seed_candidates_copy['seed_pmid'].apply(lambda x: 'pmid:' + str(int(x)) if pd.notnull(x) else x))
                seed_candidates_copy.rename(columns = {'seed_Id' : 'id'}, inplace = True)
                seed_candidate_generic_df_ss = self.handsearch_cls.retrieve_generic_paper_details(seed_candidates_copy)
                if len(seed_candidate_generic_df_ss) == len(seed_candidates_copy):
                    print('Data retrieved from api is same length as seed candidates. Updating original class: {}'.format(original_review_id))
                    seed_candidate_generic_df_ss = seed_candidate_generic_df_ss[col_keep_ss]
                    seed_candidate_generic_df_ss.rename(columns = {'referenceCount' : 'references', 'citationCount' : 'citations', 'externalIds.PubMed' : 'pmid', 'externalIds.MAG' : 'mag_id', 'id' : 'seed_Id'}, inplace = True)
                    seed_candidate_generic_df_ss['citation_limit_exceeded'] = seed_candidate_generic_df_ss['citations'].apply(lambda x : 1 if x > self.citation_limit else 0)

                    #do sorting, ensure order of id in api response is same as copy of seed_candidates
                   # Set the index for both DataFrames
                    seed_candidate_generic_df_ss.set_index('seed_Id', inplace=True)
                    seed_candidates_copy.set_index('id', inplace=True)

                    # Reorder seed_candidate_generic_df_ss to match the order of seed_candidates_copy
                    seed_candidate_generic_df_ss = seed_candidate_generic_df_ss.reindex(seed_candidates_copy.index)
                    seed_candidate_generic_df_ss.set_index(seed_candidates.index, inplace=True)

                    #find new col 
                    new_col = set(seed_candidate_generic_df_ss.columns) - set(self.seed_candidates.columns)
                    if new_col: 
                        print('New columns found: {}'.format(new_col))
                        for col in new_col: 
                            self.seed_candidates[col] = None
                        self.seed_candidates.update(seed_candidate_generic_df_ss)
                    elif not new_col:
                        print('No new columns found')
                        self.seed_candidates.update(seed_candidate_generic_df_ss)
                    
                    print('Updating seed article dataframe with generic paper details from api for original review: {}'.format(original_review_id))
                else: 
                    self.logger.warning('Data retrieved not same length as seed candidates. Please check handling') 


    def update_data_columns(self, source_df, included_article_slice):

        columns_to_update = ['title', 'abstract', 'year', 'paper_Id', 'type',
                            'citations', 'references', 'api_path', 
                            'no_data_from_api', 'no_data_no_id', 'not_retrieved']
        
        new_columns = set(columns_to_update) - set(self.included_articles.columns.to_list())

        if new_columns: 
            #creating new columns 
            for col in new_columns:
                self.included_articles[col] = None
        #perform checks on nunber of rows in source and target 
        if len(source_df) != len(included_article_slice):
            print('Length of source_df is not the same as length of target_indices. Please check')
        elif len(source_df) == len(included_article_slice):
            source_df.index = included_article_slice.index
            self.included_articles.loc[source_df.index, columns_to_update] = source_df[columns_to_update]


    def calculate_additional_columns(self, df):
        df_copy = df.copy()
        if 'no_data_from_api' in df_copy.columns:
            df_copy['no_data_from_api'].fillna(0, inplace=True)
        elif 'no_data_from_api' not in df_copy.columns:
            df_copy['no_data_from_api'] = 0
        df_copy['no_data_no_id'] = df_copy.apply(lambda x: 1 if pd.isna(x['paper_Id']) else 0, axis=1)
        df_copy['not_retrieved'] = df_copy.apply(lambda x: 1 if (x['no_data_no_id'] == 1 and pd.isna(x['no_data_from_api'])) else 0, axis=1)
            
        return df_copy
        
        


    def generate_temp_sorting_column(self,df): 
        '''function to generate temporary sorting column for included articles dataframe'''
        df_copy = df.copy()
        #generate sorting column for included articles dataframe
        if self.current_api == 'semanticscholar':  
            df_copy['temp_sorting_id'] = df_copy['included_doi'].str.lower()
            # Fill 'id' with 'included_pmid' if it is not null
            df_copy['temp_sorting_id'] = df_copy['temp_sorting_id'].fillna(
                df_copy['included_pmid'].apply(lambda x: str(int(x)) if pd.notnull(x) else NaN)
            )

            # Fill 'id' with 'included_mag_id' if it is not null
            df_copy['temp_sorting_id'] = df_copy['temp_sorting_id'].fillna(
                df_copy['included_mag_id'].apply(lambda x: str(int(x)) if pd.notnull(x) and x != 'nan' else NaN)
            )

            df_copy['temp_sorting_id'] = df_copy['temp_sorting_id'].fillna(
                df_copy['ref_if_no_id'].apply(lambda x: 'no_id: ' + x if pd.notnull(x) and x!= 'nan' else NaN)
            )

        elif self.current_api == 'openalex': 
            df_copy['temp_sorting_id'] = df_copy['id']

        return df_copy

    def intra_cluster_sim_helper(self):

        for original_review_id in self.basic_data['id']:
            included_article_data = pd.DataFrame()
            print(f'Retrieving paper details for included articles, for original review: {original_review_id}')
            included_articles = self.included_articles.query('original_sr_id == @original_review_id')
            
            
            #what seems to be happening is that included_articles increases in columns after the first pass 
            included_article_data = self.handsearch_cls.retrieve_generic_paper_details(included_articles)

            if self.current_api == 'semanticscholar': 
                rename_dict = {
                    'publicationTypes': 'type',
                    'referenceCount': 'references',
                    'citationCount': 'citations',
                    'originating_api_path': 'api_path'
                }
                included_article_data.rename(columns=rename_dict, inplace=True)

            #align index of included article data with included articles
            included_articles = self.generate_temp_sorting_column(included_articles)
            included_articles_temp_sorted = included_articles.set_index('temp_sorting_id')
            if self.current_api == 'semanticscholar':
                included_article_data.set_index('id', inplace = True)
            elif self.current_api == 'openalex':
                included_article_data.set_index('sorting_ids', inplace = True)
            # Reindex included_article_data to match the order of included_articles
            included_article_data = included_article_data.reindex(included_articles_temp_sorted.index)

            if 'abstract_inverted_index' in included_article_data.columns:
                #only relevant for openalex 
                print('Decoding abstracts.')
                included_article_data['abstract'] = self.handsearch_cls.api_interface.decode_abstract(included_article_data['abstract_inverted_index'])
            
            included_article_data = self.calculate_additional_columns(included_article_data)
            self.update_data_columns(included_article_data, included_articles)
            print(f'Generating intra-cluster similarity for original review: {original_review_id}')
            intra_cluster_sim = self.generate_intra_cluster_similarity(included_article_data[['title', 'abstract']])
            self.basic_data.loc[self.basic_data['id'] == original_review_id, 'intra_cluster_similarity'] = intra_cluster_sim
    
    def generate_intra_cluster_similarity(self, title_abstract_data_in):
        '''function to generate intra cluster similarity for a dataframe where each entry is the abstract of the included article'''
        #load sentence bert trained on pubmed model from huggingface 
        model = SentenceTransformer('allenai-specter')
        nltk.download('punkt')
        abstract_data_df = pd.DataFrame()
        #remove no abstract found and set to none
        title_abstract_data = title_abstract_data_in.copy()
        title_abstract_data['abstract'] = title_abstract_data['abstract'].apply(lambda x : None if x == 'No abstract Found' else x)
        title_abstract_data['title'] = title_abstract_data['title'].apply(lambda x : None if x == 'No title found' else x)
        #remove text enclosed in html tags 
        title_abstract_data['abstract'] = title_abstract_data['abstract'].apply(lambda x : re.sub('<.*?>', '', x) if x is not None else x)
        #remove the word "abstract"
        title_abstract_data['abstract'] = title_abstract_data['abstract'].apply(lambda x : re.sub(r'\babstract\b', '', x, flags=re.IGNORECASE) if x is not None else x)
        #add full stop to titles and add to abstract 
        title_abstract_data['title'] = title_abstract_data['title'].apply(lambda x : x + '.' if pd.isna(x) == False else x)
        title_abstract_data['title_abstract'] = title_abstract_data['title'].astype(str) + title_abstract_data['abstract'].astype(str)
        #tokenize title and abstract into sentences
        abstract_data_df['sentences'] = title_abstract_data['title_abstract'].apply(lambda x : nltk.tokenize.sent_tokenize(x))
        #convert each sentence into sentence embedding
        abstract_data_df['sentence_embedding'] = abstract_data_df['sentences'].apply(lambda x : model.encode(x))
        abstract_data_df['abstract_embedding'] = abstract_data_df['sentence_embedding'].apply(lambda x : x.mean(axis = 0))

        #compute cosine similiarity between all abstract embeddings
        abstract_embeddings = abstract_data_df['abstract_embedding'].tolist()
        intra_cluster_sim = cosine_similarity(abstract_embeddings).mean()
        return intra_cluster_sim
    
    def update_workbook_with_seed_article_data(self,df,api, original_sr_id): 
        '''
        receives dataframe containing seed article data and api used to generate data and updates workbook acordingly with new tab
        '''
        #saving character space 
        if api == 'openalex':
            api = 'oa'
        elif api == 'semanticscholar':
            api = 'ss'

        with pd.ExcelWriter(self.file_path_updated, engine="xlsxwriter", mode="a", if_sheet_exists='overlay') as writer:
            #copy seed article candidate data to new tab in workbook
            tab_name = 'seed_data_{}'.format(api)

            #find index for original_sr_id 
            original_sr_id_index = self.seed_candidates.query('original_sr_id == @original_sr_id').index
            seed_candidate_data = self.seed_candidates
            # print('Length of original SR index',len(original_sr_id_index))
            # print('Length of df',len(df))
            # seed_candidate_data.loc[original_sr_id_index,'title'] = df['title']
            # seed_candidate_data.loc[original_sr_id_index, 'year_published'] = df['year']
            # seed_candidate_data.loc[original_sr_id_index, 'seed_api_id'] = df['paper_Id']
            # seed_candidate_data.loc[original_sr_id_index, 'citations'] = df['citationCount']
            # seed_candidate_data.loc[original_sr_id_index, 'references'] = df['referenceCount']
            df_copy = (df.copy()).reset_index(drop = True)
            print(tab_name)
            df_copy.to_excel(writer, sheet_name=tab_name)
