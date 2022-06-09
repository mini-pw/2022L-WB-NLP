import json
import ast
import pickle as pkl
import pandas as pd
import numpy as np
import requests

from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

tqdm.pandas()


def get_flattened_history(response):
    response = json.loads(response.content)
    history_items = response['history'].items()
    history_columns = [f'history_{entry[0]}' for entry in history_items]
    history_values = [[entry[1]] for entry in history_items]
    return dict(zip(history_columns, history_values))


relevant_response_fields = [
    'cited_by_posts_count',
    'cited_by_tweeters_count',
    'cited_by_policies_count',
    'readers_count',
    'score',
]


def extract_relevant_fields(response):

    response = json.loads(response.content)

    return dict(zip(relevant_response_fields,
                    [[response.get(key, 0)] for key in relevant_response_fields]))


def get_altmetric_response(row):
    try:
        if row['doi']:
            response = requests.get(
                f"https://api.altmetric.com/v1/doi/{row['doi']}")
        elif row['pubmed_id']:
            response = requests.get(
                f"https://api.altmetric.com/v1/pmid/{row['pubmed_id']}")
        elif row['arxiv_id']:
            response = requests.get(
                f"https://api.altmetric.com/v1/arxiv/{row['arxiv_id']}")
        else:
            response = None
    except:
        print('Error occured')
        return None

    return response


def main():

    df = pd.read_csv('data/s2orc_ai.csv')
    df["mag_field_of_study"] = df["mag_field_of_study"].fillna("['Missing']").apply(ast.literal_eval)

    print('Filtering empty ids')

    df = df.loc[~(df['doi'].isna() & df['pubmed_id'].isna()
                  & df['arxiv_id'].isna())]

    print('Getting Altmetrics responses')
    # responses = df.progress_apply(get_altmetric_response, axis='columns')
    with open('data/altmetrics_responses.pkl', 'rb') as f:
        responses = pkl.load(f)

    print('Saving responses to pickle')
    with open('data/altmetrics_responses.pkl', 'wb') as f:
        pkl.dump(responses, f)

    print('Filtering empty responses')
    response_found = list(
        map(lambda response: response is not None and
            response.status_code == 200, responses))
    response_col = pd.DataFrame(
        {'response': np.array(responses)[response_found]})

    print('Merging responses to DataFrame')
    df_with_altmetric = pd.concat([df.loc[response_found].reset_index(drop=True), response_col],
                                  axis='columns')

    print('Extracting relevant data from reponses')
    relevant_fields = df_with_altmetric['response'].apply(
        extract_relevant_fields).to_list()
    relevant_fields_rows = list(map(pd.DataFrame, relevant_fields))
    relevant_fields_df = pd.concat(relevant_fields_rows, axis='rows')
    relevant_fields_df.index = np.arange(len(relevant_fields_df))

    history_fields = df_with_altmetric['response'].apply(
        get_flattened_history).to_list()
    history_fields_rows = list(map(pd.DataFrame, history_fields))
    history_fields_df = pd.concat(history_fields_rows, axis='rows')
    history_fields_df.index = np.arange(len(history_fields_df))

    df_with_altmetric = pd.concat(
        [df_with_altmetric, relevant_fields_df, history_fields_df], axis='columns')

    df_with_altmetric = df_with_altmetric.drop(columns=['response'])

    df_with_altmetric['mag_field_of_study'] = df_with_altmetric['mag_field_of_study'].apply(
        lambda x: ['None'] if not x else x)

    with open('data/data_with_altmetric.pkl', 'wb') as f:
        pkl.dump(df_with_altmetric, f)


if __name__ == '__main__':
    main()
