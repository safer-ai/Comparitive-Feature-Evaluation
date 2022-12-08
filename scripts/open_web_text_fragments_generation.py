#%%
# THIS WAS RUN ON THE REDWOOD CODE BASE
# from interp.tools.data_loading import get_val_seqs
# from interp.tools.interpretability_tools import toks_to_string_list
# v = get_val_seqs(n_files=1)
# s = toks_to_string_list(v[:, 1:]) # Exclude begin token
s: list[str] = []

from tqdm import tqdm  # type: ignore
import pandas as pd

fragments: list[str] = []
for ss in tqdm(s):
    # Exclude the start space, and select 32 words
    fragments.append(" ".join(ss.split(" ")[1:33]))

df = pd.DataFrame(fragments, columns=["fragment"])

df.to_csv("raw_data/open_text_fragments.csv")
#%%
# You can still run this from here to do manual feature selection
import pandas as pd

df = pd.read_csv("raw_data/open_text_fragments.csv", index_col=0)
df.head(30)
for i, row in df.iterrows():
    print("---", i, "---")
    print(row.fragment)
# %%
# Manual selection of gender-empty texts
gender_empty = set(range(25)) - set([5, 12, 14, 15, 18, 21, 23])
# Manual selection of football empty text
football_empty = set(range(25))
last_checked = 24
df["gender_empty"] = df.index.map(
    lambda i: 1 if i in gender_empty else (-1 if i <= last_checked else 0)
)
df["football_empty"] = df.index.map(
    lambda i: 1 if i in football_empty else (-1 if i <= last_checked else 0)
)
df.head(30)

# %%
df.to_csv("raw_data/open_text_fragments.csv")
# %%
