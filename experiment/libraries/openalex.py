import asyncio
import aiohttp 
import pandas as pd
from aiolimiter import AsyncLimiter
import platform 
import re 
import rispy 
from numpy import NaN 
import logging 
from collections import OrderedDict
import numpy as np

class openalex_interface: 

    '''
    Convenience class for interacting with the openalex api interface. Main functionality at the moment is to conduct snowballing / citation mining. Must provide a 
    dataframe containing a column of article ids (either DOI or OpenAlex format).
    '''

    def __init__(self): 

        self.api_limit = AsyncLimiter(3,1)
        self.api_fields = ['id','doi','title','publication_year', 'ids', 'locations', 'language', 'referenced_works', 'referenced_works_count', 'type','abstract_inverted_index', 'cited_by_api_url', 'cited_by_count']
        self.api_fields_str = ','.join(self.api_fields)
        self.pagination_limit = 200
        self.default_cursor = '*'
        #flag for different api_paths 
        self.generic = False
        self.logger = logging.getLogger(__name__)
        if platform.system()=='Windows':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    def chunk_id_list(self,id_list): 
        '''OpenAlex has a limit of 50 ids per request, which can be concatenated together with the pipe operator.
        This function takes a list containing article ids returns a list of chunks with maximum length of 50
        '''
        max_list_length = 50 
        id_chunks = [id_list[x:x+max_list_length] for x in range(0, len(id_list), max_list_length)]

        id_chunks = ['|'.join(map(str, l)) for l in id_chunks]
        return id_chunks 

    def decode_abstract(self, inverted_index_dict): 
        '''Takes the inverted index dictionary from the OpenAlex API and returns abstract in human readable form'''
        abstract_list = []
        for j in inverted_index_dict:
            if pd.isna(j):
                self.logger.warning('No abstract found for this article')
                print('No abstract found for this article')
                abstract = None
                abstract_list.append(abstract)
            else:
                self.logger.info('Decoding abstract')
                abstract_index = {v: k for k, vlist in j.items() for v in vlist}
                abstract = ' '.join(abstract_index[k] for k in sorted(abstract_index))
                abstract_list.append(abstract)
                self.logger.info('Abstract decoded.')
        return abstract_list

    def generate_default_api_path(self,id): 
        '''Checks if id is a DOI or OpenAlex ID and returns appropriate API endpoint(s). Pagination limit is set to 200, but this can
        be modified inside __init__ method if needed. Note that id is a id list, not a single id.
        '''
        
        oa_path_dict_list = []

        if self.generic == True: 
            openalex_api_endpoint = 'https://api.openalex.org/works?filter={}:{}&select={}&per-page={}&cursor={}' 

            for i in id: 
                openalex_api_path_dict = OrderedDict()
                if i.startswith('10.'):
                    openalex_api_path = openalex_api_endpoint.format('doi',i,self.api_fields_str,self.pagination_limit,self.default_cursor)
                elif i.startswith('pmid'):
                    #remove trailing pmid from i 
                    i = re.sub(r'pmid:', '', i)
                    openalex_api_path = openalex_api_endpoint.format('pmid',i,self.api_fields_str,self.pagination_limit,self.default_cursor)
                elif len(i) == 8: 
                    openalex_api_path = openalex_api_endpoint.format('pmid',i,self.api_fields_str,self.pagination_limit,self.default_cursor)
                elif i.startswith('https://openalex.org/W'):
                    openalex_api_path = openalex_api_endpoint.format('openalex',i,self.api_fields_str,self.pagination_limit,self.default_cursor)
                elif i.startswith('no_id:'): 
                    openalex_api_path = openalex_api_endpoint.format('no_id',i,self.api_fields_str,self.pagination_limit,self.default_cursor)
                else:
                    #assuming the only other alternative mag id 
                    openalex_api_path = openalex_api_endpoint.format('mag',i,self.api_fields_str,self.pagination_limit,self.default_cursor)

                openalex_api_path_dict['originating_id_chunk'] = i
                openalex_api_path_dict['api_path'] = openalex_api_path
                oa_path_dict_list.append(openalex_api_path_dict)

        elif self.generic == False: 
            #self.generic false refers to whether we are just retrieving paper details or when self.generic == False, dealing with cited_by_api_url and reference_works
            openalex_api_endpoint = 'https://api.openalex.org/works?filter={}:&select={}&per-page={}&cursor={}' 
            #check whether id list is empty in the first instance implying no reference data 
            if len(id) == 0: 
                openalex_api_path_dict = OrderedDict()
                print ('ID list seems to be empty, check whether there are any references or citations for this article')
                self.logger.warning('ID list seems to be empty, check whether there are any references or citations for the originating article')
                openalex_api_path = None
                openalex_api_path_dict['originating_id_chunk'] = None
                openalex_api_path_dict['api_path'] = openalex_api_path
                oa_path_dict_list.append(openalex_api_path_dict)

            else: 
                for i in id:  
                    openalex_api_path_dict = OrderedDict()
                    if i is None or not i:
                        print('No ID found, checking seed pmid column')
                        openalex_api_path = openalex_api_endpoint.format(i,self.api_fields_str,self.pagination_limit,self.default_cursor)
                        self.logger.info('PMID found, using PMID API endpoint. Generating API path.')
                    
                    elif i and i.startswith('10.'):
                        openalex_api_path = openalex_api_endpoint.format('doi:'+i, self.api_fields_str, self.pagination_limit,self.default_cursor)
                        self.logger.info('DOI found, using DOI API endpoint. Generating API path. ', extra={'doi': i})

                    elif i and i.startswith('https://openalex.org/W'): 
                        openalex_api_path = openalex_api_endpoint.format('openalex:'+i, self.api_fields_str, self.pagination_limit,self.default_cursor)
                        self.logger.info('OpenAlex ID found, using OpenAlex API endpoint. Generating API path. ', extra={'openalex:': i})

                    elif i and i.startswith('pmid'):
                        #remove trailing pmid from i 
                        i = re.sub(r'pmid:', '', i)
                        openalex_api_path = openalex_api_endpoint.format('pmid:'+i, self.api_fields_str, self.pagination_limit,self.default_cursor)
                        self.logger.info('PMID found, using PMID API endpoint. Generating API path. ', extra={'pmid': i})
                
                    openalex_api_path_dict['originating_id_chunk'] = i
                    openalex_api_path_dict['api_path'] = openalex_api_path
                    oa_path_dict_list.append(openalex_api_path_dict)

        return oa_path_dict_list
    
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
        #only return lists that are not empty 
        split_list = [i for i in [pmid_list, doi_list, mag_list, nan_list] if i != []]
        return split_list

    async def retrieve_references(self, article_df): 

        '''retrieves references from a given list of article IDs'''
        if 'id' in article_df.columns:
            article_df.rename(columns={'id':'seed_Id'}, inplace=True)

        references_openalex = article_df[['seed_Id','referenced_works', 'references']].copy()
        references_openalex.rename(columns={'seed_Id':'id'}, inplace=True)
        references_openalex = references_openalex.to_dict(orient='records')
        backward_snowball_tasks = []

        flat_ref_od_list = []
        for dict in references_openalex:
            # Generate the 'reference_chunks' and 'reference_api_path' key-value pairs
            dict['reference_chunks'] = self.chunk_id_list(dict['referenced_works'])
            dict['reference_api_path'] = self.generate_default_api_path(dict['reference_chunks'])
            
            
            # Check the type and length of 'reference_api_path'
            if isinstance(dict['reference_api_path'], list):
                if len(dict['reference_api_path']) == 1:
                    # If there's only one dictionary in the list, extract its 'api_path'
                    dict['reference_api_path'] = dict['reference_api_path'][0].get('api_path')
                    flat_ref_od_list.append(dict)
                elif len(dict['reference_api_path']) > 1:
                    # If there are multiple dictionaries in the list, extract 'api_path' from each
                    for inner_dict in dict['reference_api_path']:
                        _ref_path = inner_dict.get('api_path')
                        new_dict = {k : v for k, v in dict.items() if k!='reference_api_path'}
                        new_dict['reference_api_path'] = _ref_path
                        flat_ref_od_list.append(new_dict)

                            
        self.logger.info('Retrieving paper details for referenced works.')
        #using apply function to ensure api_path stays maped to originating id

        for i, od in enumerate(flat_ref_od_list): 
            extracted_keys = set(['id', 'reference_api_path'])
            _od = {k:v for k,v in od.items() if k in extracted_keys}
            backward_snowball_tasks.append(self.retrieve_paperdetails(_od))
        oa_results_references = await asyncio.gather(*backward_snowball_tasks) 
    

        # #loop through references_openalex['references_api_path'] find the api paths that are none, and update id (key name = originating_id_chunk)
        # for i, od in enumerate(references_api_path_od_list):
        #     if type(od) == list:
        #         for inner_dict in od: 
        #             inner_dict['originating_id_chunk'] = references_openalex['id'].loc[i]
        #             backward_snowball_tasks.append(self.retrieve_paperdetails(inner_dict))

        consolidated_dict = {}
        self.logger.info('Awaiting API call tasks to complete.')
        print('Retrieving reference details for all seed articles')
        #loop through oa_results_references and concat results that share the same id 
        for od in oa_results_references: 
            id = od['id']
            api_path = od['api_path']
            results = od['results']

            if id in consolidated_dict:
                 consolidated_dict[id].append(results)
            else: 
                consolidated_dict[id] = [results]
        
        oa_consolidated_reference_results = []
        for id, results in consolidated_dict.items(): 
            #concat results 
            concat_result = pd.concat(results)
            concat_dict = {'id': id, 'results': concat_result}
            oa_consolidated_reference_results.append(concat_dict)


        print('Fixing empty results')
        print('Finished reference retrieval')
        return oa_consolidated_reference_results

    async def retrieve_citations(self,article_df): 

        '''retrieve citations from a given list of article IDs. OpenAlex structure is a bit different as citation urls are their own thing'''

        #extract citation url path for each seed id / openalex id
        self.logger.info('Extracting citation url path for each seed id (openalex id)')
        if 'id' not in article_df.columns: 
            article_df.rename(columns={'seed_Id':'id'}, inplace=True)
        citation_openalex_path = article_df[['id','cited_by_api_url']].copy()

        #add pagination limit and cursor to citation url path 
        citation_openalex_path['cited_by_api_url'] = citation_openalex_path['cited_by_api_url'].apply(lambda x : x + ('&select={}&per-page={}&cursor={}').format(self.api_fields_str,self.pagination_limit,self.default_cursor) if x is not None else x)
        #quick renaming for parity with self.retrieve paper details requirement
        citation_openalex_path.rename(columns={'cited_by_api_url': 'api_path'}, inplace=True)
        #creating dummy dict and changing structure because I messed up a few modules ago.. 
        _dict = citation_openalex_path.to_dict(orient ='records', into = OrderedDict)
        citation_task_list = []
        for od in _dict:
            citation_task_list.append(self.retrieve_paperdetails(od))
       
        #create api call tasks for each citation url path
        oa_results_citations = await asyncio.gather(*citation_task_list)

        #fix empty results, replace with empty dataframe with appropriate columns

        for od in oa_results_citations: 
            if 'results' not in od.keys(): 
                print('Empty result detected, attempting fix')
                od['results'] = pd.DataFrame(columns = ['id','publication_year'])

        #replace id column with paper_Id to avoid confusion with seed_id column
        print('renaming columns')
        _ = [inner_dict['results'].rename(columns={'id': 'paper_Id'}, inplace=True) for od in oa_results_citations for inner_dict in od.values() if 'results' in inner_dict]

        #check input and output lengths 
        print('Number of input ids: ', len(article_df))
        print('Number of rows in result df: ', len(oa_results_citations))
        print('Citation Retrieval Done')
        
        return oa_results_citations
    
    async def retrieve_paperdetails(self,api_path_dict): 

        ''' Takes a dict of OpenAlex API URLs and returns OpenAlex api response as a dataframe.'''
        openalex_results_full = pd.DataFrame() 
        cursor = self.default_cursor
        template_dummy_df = pd.DataFrame(columns = ['id','title','abstract','publication_year','publication_date','authorships','host_venue','type'])

        #check whether api_path_dict is not a dict and has a list with none 
        if api_path_dict is None: 
            print('No api path found, skipping this api call')
            self.logger.warning('No api path found, skipping this api call')
            return openalex_results_dict
        elif type(api_path_dict) == list: 
            api_path_dict = api_path_dict[0]
            values = tuple(api_path_dict.values())
            #unpacking list of dicts

        elif type(api_path_dict) == OrderedDict or dict:
            values = tuple(api_path_dict.values())

        openalex_results_dict = OrderedDict()
        id, api_path = values

        #handle instances where there is no_id from the original data extraction dataframe
        if id.startswith('no_id:'): 
            print('This chunk is a no_id chunk, creating dummy dataframes')
            self.logger.warning('This chunk is a no_id chunk, creating dummy dataframes')
            id_pipe_split = id.split('|')
            openalex_results_empty = template_dummy_df.copy()
            openalex_results_empty['missing_id'] = id_pipe_split
            
            #convert to lower case
            openalex_results_empty['missing_id'] = openalex_results_empty['missing_id'].str.lower() 
            openalex_results_dict['id'] = id
            openalex_results_dict['api_path'] = api_path
            openalex_results_dict['results'] = openalex_results_empty
        
        elif api_path is None:
            print('Empty api path detected, possibly due to no references or citations or other no-data scenarios for this article: {}'.format(id))
            self.logger.warning('Empty api path detected, possibly due to no references or citations or other no-data scenarios for this article. Creating dummy dataframes')
            #create dummy result 
            openalex_results_empty = template_dummy_df.copy()
            openalex_results_dict['id'] = id
            openalex_results_dict['api_path'] = api_path
            openalex_results_dict['results'] = openalex_results_empty

        else: 
            async with self.api_limit: 
                async with aiohttp.ClientSession() as session:
                    print('Sending request')
                    async with session.get(api_path, headers={"mailto":"darren.rajit1@monash.edu"}) as resp: 
                        if resp.status != 200: 
                            self.logger.error('Inappropriate response received for path: {}, {}. {}'.format(api_path, resp.status, resp.reason))
                            print('Error from API Call')

                            if resp.status == 429: 
                                #add retry logic 
                                self.logger.warning('API limit reached, retrying in 2 seconds')
                                print('API limit reached, retrying in 2 seconds')
                                await asyncio.sleep(2)
                                api_path_dict[id] = api_path 

                        elif resp.status == 200: 
                            print('request successful')
                            self.logger.info('Response received for path: {}'.format(api_path))
                            content = await resp.json()
                            openalex_results = pd.json_normalize(content, record_path = 'results', max_level=0)
                            openalex_results['api_path'] = api_path
                            resp_meta = content.get('meta')

                            if openalex_results.empty == True or []:
                                self.logger.warning('No results for path: {}'.format(api_path))
                                #create empty dataframe with appropriate columns
                                openalex_results_dummy = pd.DataFrame(columns = ['id','title','abstract','publication_year','publication_date','authorships','host_venue','type'])
                                openalex_results_dict['id'] = id 
                                openalex_results_dict['results'] = openalex_results_dummy

                            elif openalex_results.empty == False:

                                openalex_results_full = pd.concat([openalex_results_full,openalex_results])

                                if self.generic == False and resp_meta['count'] >= self.pagination_limit: 
                                    #count number of pages for pagination 
                                    pagination_number = round(resp_meta['count'] / self.pagination_limit) 
                                    print('Pagination detected, total of {} pages for current {}'.format(pagination_number,id))
                                    self.logger.warning('Pagination detected, total of {} pages for current {}'.format(pagination_number,id))
                                    if pagination_number > 200: 
                                        print('Pagination number if way too high, skipping')
                                        self.logger.warning('Pagniation number if way too high for {}'.format(id))
                                    
                                    while cursor is not None:
                                        self.logger.info('Pagination detected, retrieving next page')
                                        async with self.api_limit:
                                            cursor = resp_meta['next_cursor']
                                            api_path = re.sub(r"(?<=cursor\=).*$",cursor, api_path)
                                            # self.logger.info('Pagination API path:', str(api_path))
                                            async with session.get(api_path, headers={"mailto":"darren.rajit1@monash.edu"}) as pagination_resp: 
                                                if pagination_resp.status == 200: 
                                                    self.logger.info('Pagination Response received for path: {}'.format(api_path))
                                                    pagination_content = await pagination_resp.json()
                                                    openalex_paginated_results = pd.json_normalize(pagination_content, record_path = 'results', max_level=0)

                                                    if openalex_paginated_results.empty == True:
                                                        print('Pagination results empty')
                                                        self.logger.warning('Pagination results are empty, perhaps due to reaching end of pagination results. Breaking loop')
                                                        openalex_results_full = pd.concat([openalex_results_full,openalex_paginated_results])
                                                        break

                                                    elif openalex_paginated_results.empty == False:
                                                        self.logger.info('Pagination results are not empty, continuing loop')
                                                        resp_meta = pagination_content.get('meta')
                                                        print('Extracting data from pagination call, for id: {}'.format(id))
                                                        openalex_results_full = pd.concat([openalex_results_full,openalex_paginated_results])
                                                        cursor = resp_meta['next_cursor'] 

                                                elif resp.status != 200: 
                                                    self.logger.error('Inappropriate received for path: {}, {}. {}'.format(api_path, resp.status, resp.reason))
                                                    print('Error from API Call')
                                                    if resp.status == 429: 
                                                        #add retry logic 
                                                        self.logger.warning('API limit reached, retrying in 2 seconds')
                                                        print('API limit reached, retrying in 2 seconds')
                                                        await asyncio.sleep(2)
                                                        api_path_dict[id] = api_path 
                                                    else: 
                                                        self.logger.warning('Unknown error, skipping this api call')
                                                        print('Unknown error, skipping this api call')
                                                        break
                                    
                                openalex_results_dict['id'] = id
                                openalex_results_dict['api_path'] = api_path
                                openalex_results_dict['results'] = openalex_results_full
                                        
                                # elif self.generic == True: 
                                #     # no need to do pagination 
                                #     openalex_results_full = openalex_results.copy() 
                                # elif self.generic == False and resp_meta['count'] <= self.pagination_limit: 
                                #     #no need pagination 
                                #     openalex_results_full = openalex_results.copy() 

                                
                                # result_list.append(openalex_results_dict)
        
                #check for empty results
                    if self.generic == True and openalex_results_full.empty == True: 
                        print('No results found for this chunk:')
                        self.logger.warning('No results found for this id chunk.')
                        if id.startswith('no_id:'): 
                            print('This chunk is a no_id chunk, creating dummy dataframes')
                            id_pipe_split = id.split('|')
                            openalex_results_empty = template_dummy_df.copy()
                            openalex_results_empty['missing_id'] = id_pipe_split
                            
                            #convert to lower case
                            openalex_results_empty['missing_id'] = openalex_results_empty['missing_id'].str.lower() 
                            openalex_results_dict['id'] = id
                            openalex_results_dict['api_path'] = api_path
                            openalex_results_dict['results'] = openalex_results_empty

        return openalex_results_dict

    async def retrieve_generic_paper_details(self,df): 
        
        generic_retrieval_tasks = []
        self.generic = True 
        #extract ids and chunk into 50 id chunks 
        id_chunks = []

        if 'seed_Id' in df.columns:
            #fill empty with pmid 
            df['seed_Id'] = df['seed_Id'].fillna(df['seed_pmid'].apply(lambda x: 'pmid:' + str(int(x)) if pd.notnull(x) else x))
            _lists = self.id_source_splitter(df['seed_Id'].tolist())
            for _list in _lists: 
                if np.nan in _list: 
                    print('NaN found in list, please check code')
                id_chunks.append(self.chunk_id_list(_list))
                #unpack chunks into a single list of lists 

        if 'id' in df.columns: 
            _lists = [df['id'].tolist()]
            for _list in _lists: 
                id_chunks.append(self.chunk_id_list(_list))

        elif 'included_doi' in df.columns:    
            #convert doi to lower case 
            df['included_doi'] = df['included_doi'].str.lower()
            #fill empty with pmid and mag id 
            df['included_doi'] = df['included_doi'].fillna(
                df['included_pmid'].apply(lambda x: 'pmid:' + str(int(x)) if not pd.isna(x) else x)
            ).fillna(
                df['included_mag_id'].apply(lambda x: str(int(float(x))) if (not pd.isna(x) and str(x) != 'nan') else x)
            )
            df['included_doi'] = df.apply(lambda x : 'no_id: '+x['ref_if_no_id'] if x['included_doi'] == 'nan' else x['included_doi'], axis=1)
            _lists = self.id_source_splitter(df['included_doi'].tolist())
            for _list in _lists: 
                #check whether there is NaN in the list
                if np.nan in _list:
                    print('NaN found in list, please check code')
                id_chunks.append(self.chunk_id_list(_list))

        paper_path_dict_list = []
        for chunk in id_chunks: 
            paper_path_dict_list.append(self.generate_default_api_path(chunk))

        for _ in paper_path_dict_list:
            if all(isinstance(item, list) for item in _):
                for paper_path_dict in _:
                    generic_retrieval_tasks.append(self.retrieve_paperdetails(paper_path_dict))
            else: 
                for paper_path_dict in _:
                    generic_retrieval_tasks.append(self.retrieve_paperdetails(paper_path_dict))
 
        #note that this outputs a list with a dictionary containing the id, api path and results
        result_list = await asyncio.gather(*generic_retrieval_tasks) 
        self.generic = False

        for od in result_list: 
            paper_details_df = od['results']
            if 'referenced_works' in paper_details_df.columns:
                paper_details_df.rename(columns={'id':'paper_Id',
                        'cited_by_count': 'citations',
                            'publication_year': 'year',
                            'referenced_works_count': 'references',
                        }, inplace=True)
                #update dataframe
                od['results'] = paper_details_df

        #concat details for each paper into combined dataframe
        #check length of result_list
        print('Checking length of result list', len(result_list))
        if len(result_list) >= 1: 
            print('Number of input ids: ', len(df))
            combined_paper_details_df = pd.concat(
                od['results'] for od in result_list
            )
            print('Number of rows in result df: ', len(combined_paper_details_df))
            #chcek shape of result df and length of input are the same (by rows)
            if combined_paper_details_df.shape[0] != len(df):
                print('Shape of result df and length of input df are not the same, check code')
                self.logger.warning('Shape of result df and length of input df are not the same, check code, possibly due to missing data from api or improper concatenation')
            elif combined_paper_details_df.shape[0] == len(df):
                print('Shape of result df and length of input df are the same, continuing')
                self.logger.info('Shape of result df and length of input df are the same, continuing')
        # combined_paper_details_df = pd.concat([inner_dict['results'] for outer_dict in result_list for inner_dict in outer_dict.values()])
        combined_paper_details_df['sorting_ids'] = combined_paper_details_df['doi'].apply(lambda x : re.sub(r'https://doi.org/', '', x) if pd.isna(x) is False else x)

        if 'included_doi' in df.columns: 
            df.rename(columns={'included_doi':'id'}, inplace=True)
        if 'seed_Id' in df.columns: 
            df.rename(columns={'seed_Id':'id'}, inplace=True)
        df['id'] = df['id'].str.lower()

        #strip trailing doi https 
        combined_paper_details_df['pmid'] = combined_paper_details_df['ids'].apply(
            lambda x: x.get('pmid').replace('https://pubmed.ncbi.nlm.nih.gov/', 'pmid:') 
            if (pd.notna(x) and pd.notna(x.get('pmid'))) else None
        )

        combined_paper_details_df['mag_id'] = combined_paper_details_df['ids'].apply(
            lambda x: x.get('mag') 
            if (pd.notna(x) and pd.notna(x.get('mag'))) else None
        )
        #check if there is a missing id between sorting ids and id
        if set(df['id']) - set(combined_paper_details_df['sorting_ids']):

            #search for missing ids in pmid and mag columns, if found replace the sorting ids with the new id 
            missing_ids = set(df['id']) - set(combined_paper_details_df['sorting_ids'])


            # Update sorting_ids for rows where pmid is in missing_ids
            mask = combined_paper_details_df['pmid'].isin(missing_ids)
            combined_paper_details_df.loc[mask, 'sorting_ids'] = combined_paper_details_df.loc[mask, 'pmid']

            # Update sorting_ids for rows where mag_id is in missing_ids
            mask = combined_paper_details_df['mag_id'].isin(missing_ids)
            combined_paper_details_df.loc[mask, 'sorting_ids'] = combined_paper_details_df.loc[mask, 'mag_id']

            # Update missing_idsm
            missing_ids = set(df['id']) - set(combined_paper_details_df['sorting_ids'])

            # if there are still missing ids fill up with the ones we know are not retrievable
            if len(missing_ids) > 0:
                print('Missing sorting ids, likely due to no doi, mag or pmid obtainable from original data extraction. Total amount: ',  len(missing_ids))
                        # Update sorting_ids for rows where doi is in missing_ids
                if 'missing_id' in combined_paper_details_df.columns:
                    print('Missing Id column detected, probably from included article ID extraction. Attempting to fill missing ids with missing_id column')
                    mask = combined_paper_details_df['missing_id'].isin(missing_ids)
                    combined_paper_details_df.loc[mask, 'sorting_ids'] = combined_paper_details_df.loc[mask, 'missing_id']
                    missing_ids_api = set(df['id']) - set(combined_paper_details_df['sorting_ids'])

                    if len(missing_ids_api) > 0:
                        print('Still Missing sorting ids, at this point most likely due to no data retrieved from api. Total amount :', len(missing_ids_api))

                    # create empty rows for missing ids in combined_paper_details_df - these are probably ids that are not covered by the api
                        missing_ids_api_list = list(missing_ids_api)
                        missing_rows = pd.DataFrame( missing_ids_api_list, columns=['sorting_ids'])
                        missing_rows['no_data_from_api'] = 1
                        combined_paper_details_df = pd.concat([combined_paper_details_df, missing_rows], axis=0)

                elif 'missing_id' not in combined_paper_details_df.columns:
                    print('No missing ID column found. This might be due to no data retrieved from api during seed article retrieval. Missing data:', missing_ids)
                    self.logger.warning('No missing ID column found. This might be due to no data retrieved from api.')
                    missing_ids_api_list = list(missing_ids)
                    missing_rows = pd.DataFrame( missing_ids_api_list, columns=['sorting_ids'])
                    missing_rows['no_data_from_api'] = 1
                    combined_paper_details_df = pd.concat([combined_paper_details_df, missing_rows], axis=0)

            
        #sort
        combined_paper_details_df_sorted = combined_paper_details_df.set_index('sorting_ids').loc[df['id']].reset_index()

        return combined_paper_details_df_sorted

    def to_ris(self, df):

        result_df_openalex = df 
        entries = result_df_openalex[['paper_Id', 'doi', 'title', 'abstract', 'publication_year', 'publication_date','authorships','host_venue', 'type']].copy()
        entries['database_provider'] = 'OpenAlex'
        
        entries.rename(columns = {'type' : 'type_of_reference'
                                ,'publication_year' : 'year'
                                ,'publication_date' : 'date'
                                ,'authorships' : 'authorship_data'
                                ,'host_venue' : 'journal_name'
                                ,'paper_Id' : 'id'
                                }, inplace = True)

        #unpacking dictionary of authors into a list of authors
        author_data = pd.json_normalize(entries['authorship_data'].apply(lambda x : eval(x)))
        author_data = author_data.applymap(lambda x: {} if pd.isnull(x) else x)


        colname_range = range(1, len(list(author_data))+1)
        new_cols = ['A' + str(i) for i in colname_range]
        author_data.columns = new_cols

        author_names = author_data.apply(lambda x : x.str.get('author.display_name'), axis = 1)
        author_names = author_names.apply(lambda x : list(x.tolist()), axis = 1)
        author_names = author_names.apply( lambda x : list(filter(lambda item: item is not None, x)))
        author_names.name = 'authors'

        entries = pd.concat([entries, author_names], axis = 1)
        entries_ris = entries.to_dict('records')
        ris_export_path = 'result.ris'
        with open (ris_export_path,'w', encoding = 'utf-8') as f:
            rispy.dump(entries_ris,f)

