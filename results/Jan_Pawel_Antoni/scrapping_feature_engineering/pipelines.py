import swifter
import spacy
import numpy as np
import pandas as pd
from collections import Counter
from functools import partial
from itertools import chain
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin


class RawDataToVec(TransformerMixin):
    def __init__(self):
        self.ap = AbstractToVecTransformer()
        self.rvfcr = RareValuesFromColumnsRemover(
            ["institutions", "journal", "authors", "countries"]
        )
        self.mlbmc = MultilabelBinarizerMulticolumn(
            [
                "institutions",
                "institutions_types",
                "authors",
                "countries",
                "mag_field_of_study",
            ]
        )
        self.ohemc = OnehotEncoderMulticolumn(["journal", "type"])
        self.df2vec = DfToFeatureVector(self.ap, self.mlbmc, self.ohemc)

    def fit(self, X, y=None, **kwargs):
        X = X.reset_index(drop=True)
        self.ap.fit(X, y)
        # Potrzebne, zeby pipeline filtrujacy zadzialal
        X["journal"] = X["journal"].fillna("No Journal")
        X["journal"] = X["journal"].apply(lambda j: [j])
        self.rvfcr.fit(X, y)
        self.mlbmc.fit(X, y)
        X, y = self.rvfcr.transform(X, y)
        X, y = self.mlbmc.transform(X, y)
        # Potrzebne, zeby kolejny pipeline zadzialal
        X["journal"] = X["journal"].apply(lambda l: l[0])
        self.ohemc.fit(X, y)
        X, y = self.ohemc.transform(X, y)
        self.df2vec.fit(X, y)
        return self

    def transform(self, X, y=None, **kwargs):
        X = X.reset_index(drop=True)
        X, y = self.ap.transform(X, y)
        # Potrzebne, zeby pipeline filtrujacy zadzialal
        X["journal"] = X["journal"].fillna("No Journal")
        X["journal"] = X["journal"].apply(lambda j: [j])
        X, y = self.rvfcr.transform(X, y)
        X, y = self.mlbmc.transform(X, y)
        # Potrzebne, zeby kolejny pipeline zadzialal
        X["journal"] = X["journal"].apply(lambda l: l[0])
        X, y = self.ohemc.transform(X, y)
        X.is_open_access = X.is_open_access.astype(int)
        X, y = self.df2vec.transform(X, y)
        return X, y


class AbstractToVecTransformer(TransformerMixin):
    def __init__(self):
        self.en = spacy.load("en_core_web_sm")

    def fit(self, X, y=None, **kwargs):
        print("Fit: converting abstract to docs")
        abstract_tokens = X["abstract"].swifter.apply(
            self._abstract_to_token_list
        )
        print("Fit: getting unique tokens from dataset")
        flatenned_tokens = list(chain(abstract_tokens))
        unique_tokens = np.unique(flatenned_tokens)
        print("Fit: calculating document frequencies for unique tokens")
        doc_freq = np.array(
            list(
                map(
                    partial(
                        self._abstracts_with_token_count,
                        abstracts_col=abstract_tokens,
                    ),
                    unique_tokens,
                )
            )
        )

        print("Fit: finding too rare and too frequent tokens")
        self.too_frequent_tokens = unique_tokens[doc_freq > 0.7 * X.shape[0]]
        self.too_rare_tokens = unique_tokens[doc_freq < 10]
        abstract_tokens = abstract_tokens.swifter.apply(
            self._drop_too_frequent_and_too_rare
        )
        print("Fit: fitting tf-idf")
        abstract_tokens = abstract_tokens.swifter.apply(
            self._merge_token_list_to_text
        )
        self.tfidf = TfidfVectorizer()
        self.tfidf.fit(abstract_tokens)

        return self

    def transform(self, X, y=None, **kwargs):
        abstract_col = X["abstract"].swifter.apply(
            self._abstract_to_token_list
        )
        abstract_col = abstract_col.swifter.apply(
            self._drop_too_frequent_and_too_rare
        )
        abstract_col = abstract_col.swifter.apply(
            self._merge_token_list_to_text
        )
        X = X.copy()
        X["abstract_encoded"] = abstract_col.swifter.apply(
            lambda l: self.tfidf.transform([l])
        )
        return X, y

    def _abstract_to_token_list(self, abstract):
        doc = self.en(abstract)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            if not token.is_punct
        ]
        return tokens

    def _abstracts_with_token_count(self, token, abstracts_col):
        is_token_inside_doc = [
            1 if token in abstract else 0 for abstract in abstracts_col.values
        ]
        return np.sum(is_token_inside_doc)

    def _drop_too_frequent_and_too_rare(self, tokens_list):
        return list(
            filter(
                lambda token: token not in self.too_frequent_tokens
                and token not in self.too_rare_tokens,
                tokens_list,
            )
        )

    def _merge_token_list_to_text(self, token_list):
        return " ".join(token_list)


class RareValuesFromColumnsRemover(TransformerMixin):
    def __init__(self, columns, threshold=2):
        self.columns = columns
        self.threshold = threshold

    def fit(self, X, y=None, **kwargs):
        self.columns_to_mapping = dict()
        for column in self.columns:
            flat = [el for l in X[column] for el in l]
            counter = Counter(flat)
            mapping = dict()
            for key, val in counter.items():
                mapping[key] = key if val >= self.threshold else "other"
            self.columns_to_mapping[column] = mapping
        return self

    def transform(self, X, y=None, **kwargs):
        X = X.copy()
        for column in self.columns:
            X[column] = X[column].swifter.apply(
                lambda l: np.unique(
                    [
                        self.columns_to_mapping[column].get(el, "other")
                        for el in l
                    ]
                )
            )
        return X, y


class MultilabelBinarizerMulticolumn(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None, **kwargs):
        self.binarizers = dict()
        for col in self.columns:
            print(f"Removing NAs from {col}")
            y = y.loc[~X[col].isna().values]
            X = X.loc[~X[col].isna()]
            mlb = MultiLabelBinarizer()
            mlb.fit(X[col])
            self.binarizers[col] = mlb
        return self

    def transform(self, X, y=None, **kwargs):
        X = X.copy()
        for col in self.columns:
            print(f"Removing NAs from {col}")
            y = y.loc[~X[col].isna().values]
            X = X.loc[~X[col].isna()]
            col_transformed = self.binarizers[col].transform(X[col])
            col_transformed = list(map(np.array, col_transformed.tolist()))
            X[f"{col}_encoded"] = col_transformed
        return X, y


class OnehotEncoderMulticolumn(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None, **kwargs):
        self.encoders = dict()
        for col in self.columns:
            print(f"Removing NAs from {col}")
            y = y.loc[~X[col].isna().values]
            X = X.loc[~X[col].isna()]
            oh = OneHotEncoder(handle_unknown="ignore")
            column = pd.DataFrame(X[col])
            oh.fit(column)
            self.encoders[col] = oh
        return self

    def transform(self, X, y=None, **kwargs):
        X = X.copy()
        for col in self.columns:
            print(f"LOOOG: {col}")
            print(f"Removing NAs from {col}")
            y = y.loc[~X[col].isna().values]
            X = X.loc[~X[col].isna()]
            column = pd.DataFrame(X[col])
            col_transformed = self.encoders[col].transform(column)
            col_transformed = list(
                map(np.array, col_transformed.todense().tolist())
            )
            X[f"{col}_encoded"] = col_transformed
        return X, y


class DfToFeatureVector(TransformerMixin):
    def __init__(self, ap, mlbmc, ohemc):
        self.columns = [
            "in_citations_count",
            "is_open_access",
            "abstract_encoded",
            "institutions_encoded",
            "institutions_types_encoded",
            "authors_encoded",
            "countries_encoded",
            "mag_field_of_study_encoded",
            "journal_encoded",
            "type_encoded",
        ]
        self.mlbmc_columns = [
            "institutions",
            "institutions_types",
            "authors",
            "countries",
            "mag_field_of_study",
        ]
        self.ohemc_columns = ["journal", "type"]
        self.ap = ap
        self.mlbmc = mlbmc
        self.ohemc = ohemc

    def fit(self, X, y=None, **kwargs):
        inv_tfidf_dict = {v: k for k, v in self.ap.tfidf.vocabulary_.items()}
        inv_tfidf_dict_items = list(inv_tfidf_dict.items())
        inv_tfidf_dict_items.sort(key=lambda x: x[0])
        self.abstract_feat_labels = [
            f"abstract_token_{item[1]}" for item in inv_tfidf_dict_items
        ]
        all_mlbmc_features = list(
            map(
                partial(self._get_features_from_mlbmc_binarizer, X=X),
                self.mlbmc_columns,
            )
        )
        self.all_mlbmc_features = [el for l in all_mlbmc_features for el in l]
        all_ohemc_features = list(
            map(
                lambda col: self.ohemc.encoders[col].get_feature_names_out(),
                self.ohemc_columns,
            )
        )
        self.all_ohemc_features = [el for l in all_ohemc_features for el in l]
        self.labels = (
            ["in_citations_count", "is_open_access"]
            + self.abstract_feat_labels
            + self.all_mlbmc_features
            + self.all_ohemc_features
        )

    def transform(self, X, y=None, **kwargs):
        abstract_col = X["abstract"].reset_index(drop=True)
        X = X[self.columns]
        X = X.apply(self._stack_row_into_vector, axis="columns")
        vals = np.stack(X.values, 0)
        df = pd.DataFrame(vals, columns=self.labels)
        df["abstract"] = abstract_col
        return df, y

    def _stack_row_into_vector(self, row):
        row["abstract_encoded"] = row["abstract_encoded"].toarray()[0]
        row["in_citations_count"] = np.array([row["in_citations_count"]])
        row["is_open_access"] = np.array([row["is_open_access"]])
        return np.concatenate(row.values.tolist(), 0)

    def _get_features_from_mlbmc_binarizer(self, colname, X):
        n = X[colname + "_encoded"][0].shape[0]
        m = np.eye(n, n)
        mlbmc_features = list(
            map(
                lambda l: l[0],
                self.mlbmc.binarizers[colname].inverse_transform(m),
            )
        )
        return list(map(lambda s: colname + "_" + s, mlbmc_features))
