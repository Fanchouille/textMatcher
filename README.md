# Text Matcher find matches based on text (TFIDF + cosine sim) between records in 2 Dfs :

# Use
See tests_to_fill for usage

See : See https://labs.yodas.com/large-scale-matrix-multiplication-with-pyspark-or-how-to-match-two-large-datasets-of-company-1be4b1b2871e
there are a few tricks to make it efficient and quick (sparse matrix + chunks)

NOTE : this is not distributed ! (non PySpark, only Numpied)


20171017 : removes some bugs (only fetched the 1st nearest and not all)
Adds disjoint set that creates unlinked clusters from paired data
Adds functionnality : create groupId & groupCol (use disjointSet & matches)
Adds functionnality : build matches only on same groups (with group_cols)



