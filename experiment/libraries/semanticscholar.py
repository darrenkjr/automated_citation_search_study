import asyncio
import aiohttp 
import pandas as pd
from aiolimiter import AsyncLimiter
import platform 
import rispy 
import re
from numpy import NaN
import logging 
import urllib.parse
from aiohttp.client_exceptions import ClientError
import random 

#test semantic scholar API functionality
# 

class semanticscholar_interface: 

    def __init__(self,api_key): 

        self.api_limit = AsyncLimiter(5,1)
        self.session_timeout = aiohttp.ClientTimeout(total=600)
        self.pagination_limit = 500
        self.default_pagination_offset = 0
        self.max_retries = 5
        self.api_key = api_key
        if api_key == 'REPLACE WITH API KEY':
            raise ValueError('No API key provided')
        self.error_log = []
        self.fields = 'title,abstract,externalIds,referenceCount,citationCount,year,publicationVenue,journal,publicationTypes'
        self.api_endpoint = 'https://api.semanticscholar.org/graph/v1/paper/{id}/{citation_direction}?offset={offset}&limit={limit}&fields={fields}'
        self.generic_paper_endpoint = 'https://api.semanticscholar.org/graph/v1/paper/{id_type}:{id}?fields={fields}'
        self.logger = logging.getLogger(__name__)
        #set logger to warning and above

        if platform.system()=='Windows':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    def generate_default_api_path(self,input_id,direction):
        
        self.logger.info('Generating default API path for id: %s and direction: %s',id,direction)
        ss_path_list = []
        #check if input_id is a doi :
        if direction == 'generic':
            self.generic = True

            for k,v in input_id.items():
                if k == 'doi_list' and v:
                    for doi in v: 
                        paper_endpoint = self.generic_paper_endpoint.format(id_type = 'doi', id = doi, fields = self.fields)
                        ss_path_list.append(paper_endpoint)
                elif k == 'pmid_list' and v: 
                    for pmid in v: 
                        paper_endpoint = self.generic_paper_endpoint.format(id_type = 'pmid', id = pmid, fields = self.fields)
                        ss_path_list.append(paper_endpoint)
                elif k == 'mag_list' and v: 
                    for mag in v: 
                        paper_endpoint = self.generic_paper_endpoint.format(id_type = 'mag', id = mag, fields = self.fields)
                        ss_path_list.append(paper_endpoint)
                elif k == 'noid_list' and v: 
                    for noid in v: 
                        paper_endpoint = self.generic_paper_endpoint.format(id_type = 'no_id', id = noid, fields = self.fields)
                        ss_path_list.append(paper_endpoint)
                else: 
                    continue

        elif direction != 'generic':
            list_dict = []
            self.generic = False 
            for i in input_id:  
                if i is None or not i: 
                    self.logger.warn('ID is None, skipping')
                    return None 
                else: 
                    api_path = self.api_endpoint.format(id =i, citation_direction = direction, offset=self.default_pagination_offset,limit =self.pagination_limit, fields = self.fields)

                    api_path_dict = {
                        'id' : i,
                        'api_path' : api_path
                    }
                
                    list_dict.append(api_path_dict)

        if direction == 'generic':
            return ss_path_list
        elif direction != 'generic':
            return list_dict
        
    async def retrieve_paper_details(self, api_path_dict):
        result = {}
        api_path, current_seed_id = self._prepare_api_call(api_path_dict)
        
        async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
            if self._is_no_id(api_path):
                processed_results = None
                return processed_results
            else:
                ss_results_json = await self._retry_api_call(session, api_path)
                processed_results = await self._process_or_dummy_results(ss_results_json, api_path,session)
            
            if self.generic:
                processed_results.rename(columns={'paperId': 'paper_Id'}, inplace=True)
                return processed_results
            else:
                if processed_results.empty: 
                    processed_results = pd.DataFrame(columns = ['id','title','abstract','year','api_path'])
                    processed_results['api_path'] = api_path
                    processed_results['no_data_from_api'] = 1
                result = {
                    'id': current_seed_id,
                    'api_path': api_path,
                    'results': processed_results
                }

                #perfor empty check for processed_results 
                if processed_results.empty:
                    print('No data found for api path: {}'.format(api_path))
                return result

    def _prepare_api_call(self, api_path_dict):
        if self.generic:
            return api_path_dict, None
        else:
            return api_path_dict['api_path'], api_path_dict['id']

    def _is_no_id(self, api_path):
        return api_path.startswith('https://partner.semanticscholar.org/graph/v1/paper/no_id') and self.generic

    async def _process_or_dummy_results(self, ss_results_json, api_path, session):
        if ss_results_json is None:
            self.logger.error(f"Failed to retrieve data for api path {api_path} after retries")
            if self.generic is True:
                dummy_data = {
                    'no_data_from_api': 1,
                    'originating_api_path': api_path
                }
                return pd.DataFrame(dummy_data, index=[0])
            elif self.direction == 'citations' or self.direction == 'references':
                dummy_df = pd.DataFrame(columns = ['id','title','abstract','year','api_path'])
                dummy_df['api_path'] = api_path
                return dummy_df 
        else:
            return await self._process_results(ss_results_json, api_path,session)
    
    async def _retry_api_call(self, session, api_path):
        retries = 0
        while retries <= self.max_retries: 
            try: 
                async with self.api_limit: 
                    async with session.get(api_path, headers = {'x-api-key':self.api_key}, ssl=False) as resp:
                        if resp.status == 200: 
                            return await resp.json() 
                        elif resp.status == 429: 
                            self.logger.warning('API limit reached, backing off')
                        elif resp.status == 504: 
                            self.logger.warning('Server-side timeout, backing off')
                        elif resp.status == 404: 
                            self.logger.warning(f'404 Error for following api path: {api_path}, probably due to no data found.')
                            error_text = await resp.json()
                            return None
                        else: 
                            self.logger.warning(f'API call failed with status code: {resp.status}, backing off')
            except ClientError as e:
                self.logger.error(f'Client side error: {e}')

            retries +=1 
            if retries > self.max_retries: 
                self.logger.error(f'Failed to retrieve data for api_path {api_path} after retries')
                return None
                    # Exponential backoff with jitter
            backoff_time = (2 ** retries) + random.uniform(0, 1)
            await asyncio.sleep(backoff_time)

    async def _process_results(self,json,api_path,session):
        if self.generic is True: 
            initial_result_df = pd.json_normalize(json)
        else: 
            initial_result_df = pd.json_normalize(json,record_path=['data'])
        data_len = len(initial_result_df)
        if data_len >= self.pagination_limit: 
            combined_data = await self._handle_pagination(initial_result_df,session, api_path)
            #pagination data should be a list of dataframes 
            full_result_df = pd.concat(combined_data, ignore_index=True)
        else: 
            #no pagination required 
            full_result_df = initial_result_df
        
        if self.generic is not True:
            if self.direction == 'citations': 
                full_result_df['reference_or_citation'] = 'citation'
                full_result_df.columns = full_result_df.columns.str.replace('citingPaper.', '',regex=True)
            elif self.direction == 'references':
                full_result_df['reference_or_citation'] = 'reference'
                full_result_df.columns = full_result_df.columns.str.replace('citedPaper.', '',regex=True)
            full_result_df.columns = full_result_df.columns.str.replace('externalIds.', '',regex=True)
            full_result_df.rename(columns = {
                'paperId' : 'paper_Id', 
            }, inplace=True)

        full_result_df['originating_api_path'] = api_path

        doi_pattern = r"(?<=paper\/)([\w\d\./]+)(?:\/references|\/citations)"
        pmid_pattern = r"pmid:(\d+)(?:\/(?:references|citations))?"
        pattern_list = [doi_pattern,pmid_pattern]
        originating_id = next((match.group(1) for pattern in pattern_list for match in re.finditer(pattern, api_path)), None)
        full_result_df['originating_seed_id'] = originating_id

        return full_result_df

    async def _handle_pagination(self,initial_data, session,api_path):

        combined_data = [initial_data]
        pagination_offset = self.default_pagination_offset

        while True: 
            pagination_offset += self.pagination_limit
            new_api_path = re.sub(r"(?<=offset=)(.*)(?=&limit)",str(pagination_offset),api_path)

            pagination_json_data = await self._retry_api_call(session, new_api_path)
            paginated_df = pd.json_normalize(pagination_json_data,record_path=['data'])

            if not paginated_df.empty and len(paginated_df) >= self.pagination_limit: 
                combined_data.append(paginated_df)
            else: 
                if not paginated_df.empty: 
                    combined_data.append(paginated_df)
                break 
        return combined_data


    async def retrieve_citations(self, article_df): 
        '''retrieves citation data from a given article dataframe'''
        self.direction = 'citations'
        forward_snowball_tasks = []

        article_df['seed_Id'].fillna('pmid:'+ article_df['seed_pmid'], inplace=True)

        if type(article_df['seed_Id']) == str:
            id_list = [article_df['seed_Id']]
        else: 
            id_list = article_df['seed_Id'].tolist()
    
        api_path_list = self.generate_default_api_path(id_list,'citations')
        for api_dict in api_path_list:
            forward_snowball_tasks.append(self.retrieve_paper_details(api_dict))
        ss_results_citations = await asyncio.gather(*forward_snowball_tasks)
        print('Citation retrieval done')
        #returns list of dataframes 
        return ss_results_citations 
    
    async def retrieve_references(self, article_df): 
        '''retrieves reference data from a given article dataframe'''
        self.direction = 'references'
        backward_snowball_tasks = []
        ss_results_references = pd.DataFrame()
        #fill empty spots with seed pmid 
        article_df['seed_Id'].fillna('pmid:'+ article_df['seed_pmid'], inplace=True) 

        if type(article_df['seed_Id']) == str:
            id_list = [article_df['seed_Id']]
        else: 
            id_list = article_df['seed_Id'].tolist()
            
        api_path_list = self.generate_default_api_path(id_list,'references')

        for api_dict in api_path_list: 
            backward_snowball_tasks.append(self.retrieve_paper_details(api_dict))
        ss_results_references = await asyncio.gather(*backward_snowball_tasks)
        # ss_consolidated_references = pd.concat(ss_results,ignore_index=True)
        print('reference retrieval done')
        return ss_results_references
    
    async def retrieve_generic_paper_details(self,df): 
        generic_retrieval_tasks = [] 
        df_copy = df.copy()
        if 'seed_Id' in df_copy.columns:
            df['id'] = df['seed_Id'].fillna(df['seed_pmid'].apply(lambda x: 'pmid:' + str(int(x)) if pd.notnull(x) else x))
        elif 'included_doi' in df_copy.columns:
            df_copy['id'] = df_copy['included_doi'].str.lower()
            # Fill 'id' with 'included_pmid' if it is not null
            df_copy['id'] = df_copy['id'].fillna(
                df['included_pmid'].apply(lambda x: str(int(x)) if pd.notnull(x) else NaN)
            )

            # Fill 'id' with 'included_mag_id' if it is not null
            df_copy['id'] = df_copy['id'].fillna(
                df['included_mag_id'].apply(lambda x: str(int(x)) if pd.notnull(x) and x != 'nan' else NaN)
            )

            df_copy['id'] = df_copy['id'].fillna(
                df['ref_if_no_id'].apply(lambda x: 'no_id: ' + x if pd.notnull(x) and x!= 'nan' else NaN)
            )
    
        id_list = df_copy['id'].tolist()
        id_dict = self.id_source_splitter(id_list)
        api_path = self.generate_default_api_path(id_dict,"generic")
        self.logger.info('Finished generating API paths for generic paper details')
        for url in api_path: 
            # self.logger.info('Appending generic paper details retrieval task for the following API endpoint: ', url)
            generic_retrieval_tasks.append(self.retrieve_paper_details(url))
        
        self.logger.info('Awaiting generic paper details retrieval tasks to complete')
        ss_results_list = await asyncio.gather(*generic_retrieval_tasks)
        ss_results = pd.concat(ss_results_list, ignore_index= True)

        if 'original_sr_id' in df_copy.columns: 
            #identifies that current operation is either retrieving included or seed article details 
            ss_results['input_id'] = ss_results.apply(self.extract_id, axis = 1)
            df_copy['encoded_id'] = df_copy['id'].apply(lambda x : urllib.parse.quote(x))
            if 'included_doi' in df_copy.columns: 
                target_col_merge = ['original_sr_id','id','encoded_id','ref_if_no_id', 'not_retrieved']
            else: 
                col_exclude = ['citations','references']
                target_col_merge = df_copy.columns.difference(col_exclude).tolist()
            
            print(df_copy['encoded_id'])
            ss_results_merge = pd.merge(df_copy[target_col_merge], ss_results, left_on = 'encoded_id', right_on='input_id', how = 'left', suffixes = ('_input', '_ssresp'))
            ss_results_merge.drop(columns = ['encoded_id'], inplace = True)
            rename_dct = {} 
            drop_col = []
            for col in ss_results_merge.columns: 
                if col.endswith('_ssresp'):
                    api_col_name = col.replace('_ssresp','')
                    rename_dct[col] = api_col_name 
                    drop_col.append(api_col_name + '_input')
            ss_results_merge.rename(columns = rename_dct, inplace= True)
            ss_results_merge.drop(columns = drop_col, inplace = True)

            if 'no_data_from_api' not in ss_results_merge.columns:
                ss_results_merge['no_data_from_api'] = 0
            else: 
                ss_results_merge['no_data_from_api'].fillna(0, inplace = True)

            ss_results_merge['title'] = ss_results_merge['title'].fillna('No title found')
            ss_results_merge['abstract'] = ss_results_merge['abstract'].fillna('No abstract found')

            ss_results = ss_results_merge.copy()

        return ss_results

    def extract_id(self, row):
        match = re.search(r'(?:doi|pmid|mag):([^?]+)', row['originating_api_path'])
        if match:
            return match.group(1)
        else:
            return NaN 
    
    def id_source_splitter(self,id_list): 
        '''Checks id list for mixed formats, such as DOI vs PMID and splits it into separate lists'''

        pmid_list = []
        doi_list = []
        mag_list = []
        nan_list = []

        for item in id_list:
            if item.startswith('no_id:'): 
                self.logger.warning('no_id found in id list - probably due to unobtainable DOI / PMID or MAG during data extraction')
                nan_list.append(item)
            elif item.startswith('pmid'):
                pmid_list.append(item)
            elif item.startswith('10.'):
                doi_list.append(item)
            else: 
                mag_list.append(item)

        doi_list_encoded = [urllib.parse.quote(doi) for doi in doi_list]

        #only return lists that are not empty 
        list_dict = {
            'doi_list' : doi_list_encoded,
            'pmid_list' : pmid_list,
            'mag_list' : mag_list,
            'noid_list' : nan_list
        }

        return list_dict

    def to_ris(self,df): 

        result_df_ss = df 
        entries = result_df_ss.copy() 
        entries['database_provider'] = 'Semantic Scholar'
        entries.rename(columns ={
            'paper_Id':'id',
            'paper_Title':'title',
            'paper_Abstract':'abstract',
            'paper_Venue':'journal_name',
            'paper_Year':'year',
            'paper_author':'authors',
        }, inplace=True)

        #unpack author column to get list of authors (nested dictionary)
        author_data = pd.json_normalize(entries['authors'].apply(lambda x : eval(x)))
        author_data = author_data.applymap(lambda x: {} if pd.isnull(x) else x)
        colname_range = range(1, len(list(author_data))+1)
        new_cols = ['A' + str(i) for i in colname_range]
        author_data.columns = new_cols
        author_names = author_data.apply(lambda x : x.str.get('name'), axis = 1)
        author_names = author_names.apply(lambda x : list(x.tolist()), axis = 1)
        author_names = author_names.apply( lambda x : list(filter(lambda item: item is not None, x)))
        author_names.name = 'authors'
        entries = pd.concat([entries, author_names], axis = 1)
        entries_ris = entries.to_dict('records')
        ris_export_path = 'result.ris'
        with open (ris_export_path, 'w', encoding = 'utf-8') as f: 
            rispy.dump(entries_ris,f)

        

