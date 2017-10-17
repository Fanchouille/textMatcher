import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


class TextMatcher:
    # See https://labs.yodas.com/large-scale-matrix-multiplication-with-pyspark-or-how-to-match-two-large-datasets-of-company-1be4b1b2871e
    def __init__(self, input_dfs, text_cols, id_cols, analyzer='word', ngram_range=(1, 1), stop_words=None,
                 max_features=5000):
        """
        Initialize :
        input_dfs : tuple of 2 dataframes (iDf1, iDf2) - DFs with id & text cols => we want to match items based on text cols
        text_cols : tuple of 2 strings (iTextCol1, iTextCol2) - names of cols containing the text to match on
        id_cols : tuple of 2 strings (iIdCol1, iIdCol2) - names of cols containing IDs
        stop_words : if not None, will filter out stop words before matching
        """
        self.input_dfs = input_dfs
        self.text_cols = text_cols
        self.id_cols = id_cols
        self.stop_words = stop_words
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.max_features = max_features
        return

    def get_vocabulary(self):
        """
        Concatenate all texts and create vocabulary from unique
        :return:
        """
        vect = CountVectorizer(stop_words=self.stop_words, analyzer=self.analyzer, ngram_range=self.ngram_range,
                               max_features=self.max_features)
        all_texts = np.unique(np.stack((self.input_dfs[0].loc[:, self.text_cols[0]].values,
                                        self.input_dfs[1].loc[:, self.text_cols[1]].values)).flatten())
        vocabulary = vect.fit(all_texts).vocabulary_
        self.vocabulary = vocabulary
        return

    def get_tfidf_matrices(self):
        """
        Creates the 2 TFIdf matrices
        :return:
        """
        self.get_vocabulary()
        tfidf_vect = TfidfVectorizer(stop_words=self.stop_words, vocabulary=self.vocabulary, analyzer=self.analyzer,
                                     ngram_range=self.ngram_range)
        XTf1 = tfidf_vect.fit_transform(self.input_dfs[0].loc[:, self.text_cols[0]].values)
        XTf2 = tfidf_vect.fit_transform(self.input_dfs[1].loc[:, self.text_cols[1]].values)
        self.tf_matrices = (XTf1, XTf2)
        return

    def parallelize_matrix(self, scipy_mat, rows_per_chunk):
        """
        :param scipy_mat: Scipy Matrix
        :param rows_per_chunk: # of rows per chunk
        :return: Creates chunks of matrix from a big one : with index / sparse sub matrix / shape
        """
        [rows, cols] = scipy_mat.shape
        i = 0
        submatrices = []
        while i < rows:
            current_chunk_size = min(rows_per_chunk, rows - i)
            submat = scipy_mat[i:i + current_chunk_size]
            submatrices.append((i,
                                (submat.data, submat.indices, submat.indptr),
                                (current_chunk_size, cols)
                                )
                               )
            i += current_chunk_size
        return submatrices

    def find_matches_in_submatrix(self, sources, targets, inputs_start_index, threshold):
        """
        :param sources:
        :param targets:
        :param inputs_start_index:
        :param threshold:
        :return:
        """
        cosimilarities = cosine_similarity(sources, targets)
        for i, cosimilarity in enumerate(cosimilarities):
            cosimilarity = cosimilarity.flatten()
            # Find the best matches
            target_index = np.where(cosimilarity >= threshold)[0]
            source_index = inputs_start_index + i
            if target_index.shape[0] > 0:
                for idx in target_index:
                    similarity = cosimilarity[idx]
                    if (source_index != idx):
                        yield (source_index, idx, similarity)

    def get_results(self, rows_per_chunk=100, threshold=0.8):
        """

        :param rows_per_chunk: # of rows per chunk
        :param threshold: 1 for a perfect match
        :return: a DF with 1st Id / 1st Text / 2nd Id / 2nd Text / Cosine Sim
        """
        self.get_tfidf_matrices()
        all_matrices = self.parallelize_matrix(self.tf_matrices[0], rows_per_chunk)

        results_generator = (self.find_matches_in_submatrix(csr_matrix(matrix[1], shape=matrix[2]),
                                                            self.tf_matrices[1],
                                                            matrix[0],
                                                            threshold=threshold)
                             for matrix in all_matrices)

        nearest_frn = []
        for res in results_generator:
            for x in res:
                nearest_frn.append((self.input_dfs[0].loc[:, self.id_cols[0]].iloc[x[0]],
                                    self.input_dfs[1].loc[:, self.id_cols[1]].iloc[x[1]],
                                    self.input_dfs[0].loc[:, self.text_cols[0]].iloc[x[0]],
                                    self.input_dfs[1].loc[:, self.text_cols[1]].iloc[x[1]],
                                    x[2]
                                    )
                                   )

        if self.id_cols[0] == self.id_cols[1]:
            id_col_1 = self.id_cols[0] + '_1'
            id_col_2 = self.id_cols[0] + '_2'
        else:
            id_col_1 = self.id_cols[0]
            id_col_2 = self.id_cols[1]

        if self.text_cols[0] == self.text_cols[1]:
            text_col_1 = self.text_cols[0] + '_1'
            text_col_2 = self.text_cols[0] + '_2'
        else:
            text_col_1 = self.text_cols[0]
            text_col_2 = self.text_cols[1]

        lCols = [id_col_1, id_col_2, text_col_1, text_col_2, 'COSINE_SIM']
        oDf = pd.DataFrame(nearest_frn, columns=lCols)

        return oDf
