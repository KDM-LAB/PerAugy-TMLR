import pandas as pd
from perturbation import perturbate_trajectory

aug_df = pd.read_csv("./Data/synthetic-original-augmented.csv")

news_df = pd.read_csv("./Data/PENS/news_min.tsv", sep="\t")

summ_df = pd.read_csv("./Data/summ.csv")

summ_df["PertubedSummary"] = ""

news_df.set_index('News ID', inplace=True)
summ_df.set_index('SummID', inplace=True)

perturbate_trajectory(aug_df.iloc[0], news_df, summ_df, previous_doc_steps=30, decay_rate=0.5)