import requests
import pickle as pkl
import pandas as pd
from tqdm import tqdm


def main():
    df = pd.read_csv('../data/sampled.csv')

    responses = []
    for el in tqdm(df.iterrows()):
        row = el[1]
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
        responses.append(response)

    with open('../data/responses.pkl', 'wb') as f:
        pkl.dump(responses, f)


if __name__ == '__main__':
    main()
