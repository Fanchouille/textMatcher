import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


class TextMatcher:
    def __init__(self, input_dfs, text_cols, id_cols, stop_words=None):
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
        return

    def get_vocabulary(self):

        """
        Concatenate all texts and create vocabulary from unique
        """
        vect = CountVectorizer(self.stop_words)
        all_texts = np.unique(np.stack((self.input_dfs[0].loc[:, self.text_cols[0]].values,
                                        self.input_dfs[1].loc[:, self.text_cols[1]].values)).flatten())
        vocabulary = vect.fit(all_texts).vocabulary_
        self.vocabulary = vocabulary
        return

    def get_tfidf_matrices(self):
        self.get_vocabulary()
        tfidf_vect = TfidfVectorizer(stop_words=self.stop_words, vocabulary=self.vocabulary)
        XTf1 = self.input_dfs[0].loc[:, self.text_cols[0]].values
        XTf2 = self.input_dfs[1].loc[:, self.text_cols[1]].values
        return

    def find_matches_in_submatrix(self, sources, targets, inputs_start_index, threshold=.9):
        cosimilarities = cosine_similarity(sources, targets)
        for i, cosimilarity in enumerate(cosimilarities):
            cosimilarity = cosimilarity.flatten()
            # Find the best match by using argsort()[-1]
            target_index = cosimilarity.argsort()[-1]
            source_index = inputs_start_index + i
            similarity = cosimilarity[target_index]
            if (cosimilarity[target_index] >= threshold) & (source_index != target_index):
                yield (source_index, target_index, similarity)

    def parallelize_matrix(self, scipy_mat, rows_per_chunk=100):
        """
        Creates chunk of matrix from a big one : with index / sparse sub matrix / shape
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

    def get_results(self):

    all_matrices = parallelize_matrix(X_tf)

    results_generator = (find_matches_in_submatrix(csr_matrix(matrix[1],
                                                              shape=matrix[2]),
                                                   X_tf,
                                                   matrix[0],
                                                   threshold=.9) for matrix in all_matrices)

    nearest_frn = []
    for res in results_generator:
        for x in res:
            nearest_frn.append((filetab.loc[:, 'CODE_FOUR'].iloc[x[0]],
                                filetab.loc[:, 'CODE_FOUR'].iloc[x[1]],
                                filetab.loc[:, 'L_FOUR_CLEANED'].iloc[x[0]],
                                filetab.loc[:, 'L_FOUR_CLEANED'].iloc[x[1]]))

            pd.DataFrame(nearest_frn, columns=['CODE_ORIG', 'CODE_LINKED', 'L_FOUR_ORIG', 'L_FOUR_LINKED'])
