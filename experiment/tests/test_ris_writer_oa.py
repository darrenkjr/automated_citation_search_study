import sys
sys.path.append('../automated_evidence_retrieval_study')
import rispy
import pandas as pd 
import os 
import json
from libraries.openalex import openalex_interface

file_path = os.path.join(os.path.dirname(__file__),'test_data/openalex_backwardsnowball_test.csv')
result_df_openalex = pd.read_csv(file_path)
print(result_df_openalex['paper_Id'])

openalex_interface_cls = openalex_interface()
openalex_interface_cls.to_ris(result_df_openalex)
