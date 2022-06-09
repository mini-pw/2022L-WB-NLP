import pandas as pd
import requests
import json
import numpy as np
import pickle as pkl
from tqdm.auto import tqdm
tqdm.pandas()

def get_openalex_attributes(response_content):
    try:
        result_dict = {}
        result_dict['type'] = response_content['type']
        result_dict['host_display_name'] = response_content['host_venue']['display_name']
        result_dict['publisher'] = response_content['host_venue']['publisher']
        result_dict['is_open_access'] = response_content['open_access']['is_oa']
        result_dict['open_alex_citations_count'] = response_content['cited_by_count']
        inst_nested_list = [el['institutions']
                            for el in response_content['authorships']]
        inst_nested_names = list(map(lambda inst_list: [
            entry['display_name'] for entry in inst_list], inst_nested_list))
        inst_nested_types = list(map(lambda inst_list: [
            entry['type'] for entry in inst_list], inst_nested_list))
        inst_nested_countries = list(map(lambda inst_list: [
            entry['country_code'] for entry in inst_list], inst_nested_list))
        result_dict['institutions'] = [
            el for subl in inst_nested_names for el in subl]
        result_dict['afiliation_types'] = [
            el for subl in inst_nested_types for el in subl]
        result_dict['countries'] = [
            el for subl in inst_nested_countries for el in subl]
        authorships = response_content['authorships']
        authors_list = [author['author']['display_name'] for author in authorships]
        result_dict['authors'] = authors_list
    except:
        print('Error occured')
        return None

    return result_dict

def get_openalex_raw_response(doi: str):
    url = f'https://api.openalex.org/works/doi:{doi}'
    try:
        resp = requests.get(url, timeout=10)
        content = json.loads(resp.content)
    except:
        print('Error occured')
        return None
    return content

def main():
    print('Reading data')
    df = pd.read_pickle('data/data_with_altmetric.pkl')

    print('Getting OpenAlex data')
    
    # print("Getting from API")
    # responses = df['doi'].progress_apply(get_openalex_raw_response)
    # print('Saving OpenAlex raw responses to pkl')
    # with open('data/openalex_raw_responses.pkl', 'wb') as f:
    #     pkl.dump(responses, f)

    print("Reading from file")
    with open('data/openalex_raw_responses.pkl', 'rb') as f:
        responses = pkl.load(f)

    print('Filtering null OpenAlex data')
    df = df.loc[~responses.isna()].reset_index(drop=True)
    responses = responses.loc[~responses.isna()]

    print('Extracting features from openalex responses')
    responses = responses.apply(get_openalex_attributes)

    print('Converting OpenAlex data to DataFrame')
    inst_col = responses.apply(lambda resp: resp.pop('institutions'))
    inst_type_col = responses.apply(lambda resp: resp.pop('afiliation_types'))
    authors_col = responses.apply(lambda resp: resp.pop('authors'))
    countries_col = responses.apply(lambda resp: resp.pop('countries'))

    openalex_rows = responses.apply(pd.DataFrame, index=[0]).tolist()
    openalex_df = pd.concat(openalex_rows).reset_index(drop=True)

    openalex_df = openalex_df.assign(
        institutions=inst_col.reset_index(drop=True),
        institutions_types=inst_type_col.reset_index(drop=True),
        authors = authors_col,
        countries = countries_col,
    )

    openalex_df = openalex_df.reset_index(drop=True)
    openalex_df['institutions'] = openalex_df['institutions'].apply(np.unique)
    
    openalex_df['institutions_types'] = openalex_df['institutions_types'].apply(lambda l: [el for el in l if el is not None])
    openalex_df['institutions_types'] = openalex_df['institutions_types'].apply(np.unique)

    openalex_df['authors'] = openalex_df['authors'].apply(lambda l: [el for el in l if el is not None] if type(l) is list else [])
    openalex_df['authors'] = openalex_df['authors'].apply(np.unique)

    openalex_df['countries'] = openalex_df['countries'].apply(lambda l: [el for el in l if el is not None] if type(l) is list else [])
    openalex_df['countries'] = openalex_df['countries'].apply(np.unique)

    print("Droping redundant columns")
    df = df.drop(columns=['authors'])

    print("Merging DataFrames")
    df = pd.concat([df, openalex_df], axis='columns')
    
    print("First 10 rows")
    print(df.head(10))

    print("Saving data frame to csv")
    with open('data/df_all.pkl', 'wb') as f:
        pkl.dump(df, f)

if __name__ == '__main__':
    main()
