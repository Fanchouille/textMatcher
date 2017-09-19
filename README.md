# Text Matcher find matches based on text (TFIDF + cosine sim) between records in 2 Dfs :

# Use
txt_match = TextMatcher( inputs_df = (iDf1, iDf2),
             text_cols = (TextCol1 , TextCol2)
             id_cols = (iDCol1, idCol2)
            )

iDfx = DataFrame with at least 1 id col and 1 text col
TextColx = name of the col containing text
iDCol1 = name of the col containing id

txt_match.get_results() returns a DF with matched records and the cosine sim metric.

See : See https://labs.yodas.com/large-scale-matrix-multiplication-with-pyspark-or-how-to-match-two-large-datasets-of-company-1be4b1b2871e
there are a few tricks to make it efficient and quick (sparse matrix + chunks)




