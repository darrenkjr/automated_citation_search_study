import pandas as pd
import asyncio
import os 
from dotenv import load_dotenv
from libraries.openalex import openalex_interface
from libraries.semanticscholar import semanticscholar_interface
import logging 
import numpy as np

class automated_handsearch: 

    def __init__(self,api): 

        self.logger = logging.getLogger(__name__)
        self.api = api

        if api == 'semanticscholar':
            load_dotenv() 
            semanticscholar_api_key =  os.getenv('semantic_scholar_api_key')
            self.api_interface = semanticscholar_interface(semanticscholar_api_key)

        if api == 'openalex':
            self.api_interface = openalex_interface()

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def process_rename_results(self, reference_df, citation_df, seed_id, year_limit_start, year_limit_end):
        reference_df['reference_or_citation'] = 'reference'
        citation_df['reference_or_citation'] = 'citation'
        reference_df['corresponding_seed_id'] = seed_id
        citation_df['corresponding_seed_id'] = seed_id

        if 'publication_year' in reference_df.columns:
            reference_df.rename(columns={'publication_year': 'year'},inplace=True)
        if 'id' in reference_df.columns:
            reference_df.rename(columns={'id': 'paper_Id'},inplace=True)
        
        if 'publication_year' in citation_df.columns:
            citation_df.rename(columns={'publication_year': 'year'},inplace=True)
        if 'id' in citation_df.columns:
            citation_df.rename(columns={'id': 'paper_Id'},inplace=True)

        print('Filtering by year limits')

        #perfor empty dataframe check 
        if reference_df.empty: 
            print('Reference dataframe is empty')
            #check if column exists for filter 
            if 'year' in reference_df.columns:
                print('Year column exists')
        if citation_df.empty:
            print('Citation dataframe is empty')


    # Filtering based on available year limits
        if pd.notna(year_limit_start) and pd.notna(year_limit_end):
            reference_df_filter = reference_df.query('@year_limit_start <= year <= @year_limit_end')
            citation_df_filter = citation_df.query('@year_limit_start <= year <= @year_limit_end')
        elif pd.isna(year_limit_start) and pd.notna(year_limit_end):
            reference_df_filter = reference_df.query('year <= @year_limit_end')
            citation_df_filter = citation_df.query('year <= @year_limit_end')
        elif pd.notna(year_limit_start) and pd.isna(year_limit_end):
            reference_df_filter = reference_df.query('@year_limit_start <= year')
            citation_df_filter = citation_df.query('@year_limit_start <= year')
        else: 
            print('No year limits specified')
            reference_df_filter = reference_df
            citation_df_filter = citation_df
            
        return reference_df_filter, citation_df_filter

    
    def run_citation_search(self, article_df,year_limit_start,year_limit_end): 

        '''
        Runs citation search given a dataframe containing seed article details.
        '''
        article_df_copy = article_df.copy() 
        #check if seed_Id is null, if so, replace with seed_pmid
        print('Retrieving citations')
        citations = self.loop.run_until_complete(self.api_interface.retrieve_citations(article_df_copy))
        print('Retrieving references')
        references = self.loop.run_until_complete(self.api_interface.retrieve_references(article_df_copy))
        

        # if self.api == 'openalex':
        references_list = []
        citations_list = []

        # Create a master list of unique seed IDs
        seed_id_set = set(ref_dict.get('id') for ref_dict in references if ref_dict.get('id') is not None) | \
            set(cit_dict.get('id') for cit_dict in citations if cit_dict.get('id') is not None)

        seed_id_list = list(seed_id_set)

        for seed_id in seed_id_list:
            # Append either a DataFrame or an empty DataFrame if no match is found
            references_list.append(next((ref_dict['results'] for ref_dict in references if ref_dict['id'] == seed_id), pd.DataFrame()))
            citations_list.append(next((cit_dict['results'] for cit_dict in citations if cit_dict['id'] == seed_id), pd.DataFrame()))

        # elif self.api == 'semanticscholar':
        #     #do some sorting here 
        #     references_list = references
        #     citations_list = citations

        concat_list = []

        for seed_id, (reference_df, citation_df) in zip(seed_id_list, zip(references_list, citations_list)):
            # Add a 'type' column to specify whether it's a reference or citation
            reference_df, citation_df = self.process_rename_results(reference_df, citation_df, seed_id, year_limit_start, year_limit_end)
            col_keep = ['paper_Id', 'reference_or_citation', 'year','corresponding_seed_id']
            # Subset the DataFrames based on existing columns
            reference_subset = reference_df[[col for col in col_keep if col in reference_df.columns]]
            citation_subset = citation_df[[col for col in col_keep if col in citation_df.columns]]
            # Combine the dataframes
            combine_df = pd.concat([reference_subset, citation_subset], ignore_index=True)
            
            # Check if the combined dataframe is empty
            if combine_df.empty:
                print('Both citation and reference dataframes are empty, generating empty citation network dataframe')
                combine_df = pd.DataFrame(columns=['paper_Id', 'reference_or_citation', 'year','corresponding_seed_id'])
            else:
                print('Dataframes combined successfully')
                   
            #add combined citation network to list 
            concat_list.append(combine_df)
            #size of citation network 
            citation_network_size = len(combine_df)
            print('Citation network size: {}'.format(citation_network_size))


        #convert to dataFrame 
        # results_full = pd.DataFrame(concat_list)
        print('Citation network generation done')
        #this should return a list of dataframes 
        return concat_list

    def to_ris(self,df): 
        #convert to ris format 
        ris = asyncio.run(self.api_interface.to_ris(df))
        return ris
    
    async def retrieve_generic_paper_details_async(self, df):
        return await self.api_interface.retrieve_generic_paper_details(df)
    
    def retrieve_generic_paper_details(self, df):
        # Use get_event_loop() instead of asyncio.run()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.retrieve_generic_paper_details_async(df))

    