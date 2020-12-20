import pandas as pd

df = pd.read_csv(
    "AC_ratings_google3m_koeper_SiW.csv", error_bad_lines=False, delimiter="\t"
)

train_df = df.sample(n=50000)
val_df = df.sample(n=2500)

train_df.to_csv("train/AC_ratings_google3m_koeper_SiW.csv", sep="\t")
val_df.to_csv("val/AC_ratings_google3m_koeper_SiW.csv", sep="\t")
