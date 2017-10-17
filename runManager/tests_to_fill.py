from textMatcher import textMatcher
from textMatcher import disjointSet
import pandas as pd


# Load data
# df1 =
# df2 =


# Test on different DFs
# txtMatcher = textMatcher.TextMatcher(input_dfs=(df1, df2), text_cols = (u'text', u'text2'), id_cols = ('id', 'id2'))
# res_df = txtMatcher.get_results(threshold=0.9)


# Test on same DF
# txtMatcher = textMatcher.TextMatcher(input_dfs=(df1, df1), text_cols = (u'text', u'text'), id_cols = ('id', 'id'))
# res_df = txtMatcher.get_results(threshold=0.9)


# Test on same DF with group col
# txtMatcher2 = textMatcher.TextMatcher(input_dfs=(df1, df1),
# text_cols=(u'text', u'text'), id_cols=('id', 'id'),
# group_cols=('cat', 'cat'))
# res_df2 = txtMatcher2.get_results(threshold=0.9)


# Test of disjointSet (on the same Df, with results for same DF)
# x = disjointSet.DisjointSet()
# for ft in res_df2.loc[:, 'id_1_id_2_tuple'].unique():
# st = list(ft)
# x.add(st[0], st[1])

# groupList = x.group.values() # List of unlinked groups
# print(len(groupList))


# Test on same DF with group col and add_group parameter
# txtMatcher2 = textMatcher.TextMatcher(input_dfs=(df1, df1),
# text_cols=(u'text', u'text'), id_cols=('id', 'id'),
# group_cols=('cat', 'cat'))
# res_df2 = txtMatcher2.get_results(threshold=0.9, add_groups= True)
