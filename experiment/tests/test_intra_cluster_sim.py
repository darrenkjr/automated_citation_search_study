import sys 
sys.path.append('../automated_evidence_retrieval_study')
from libraries.review_preparation import original_review
from libraries.automated_handsearch import automated_handsearch

original_review_instance = original_review('review_data/')



api_choice = 'openalex'


for original_review_id in original_review_instance.basic_data['id']: 

    #retrieve included data 
    included_articles = original_review_instance.included_articles.query('original_sr_id == @original_review_id').reset_index(drop=True)
    handsearch_cls = automated_handsearch(api_choice)
    print('Retrieving generic paper details for included articles.')
    included_article_data = handsearch_cls.retrieve_generic_paper_details(included_articles)
    abstract_inverted_index = included_article_data['abstract_inverted_index']
    print('Decoding abstracts.')
    abstract_data = handsearch_cls.api_interface.decode_abstract(abstract_inverted_index)
    included_article_data.drop(columns=['abstract_inverted_index'], inplace=True)
    included_article_data['abstract'] = abstract_data



    #generate intra cluster similarity
    print('generating intra cluster similarity for original review: {}'.format(original_review_id))
    original_review_instance.generate_intra_cluster_similarity(included_article_data[['title','abstract']])
    intra_cluster_sim = original_review_instance.intra_cluster_sim
    print('Average cosine similiarity for included articles in original review: {} is {}'.format(original_review_id,intra_cluster_sim))

