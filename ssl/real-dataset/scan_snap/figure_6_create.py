import pickle
tbl_file = "figure_6_data.pkl"
data = pickle.load(open(tbl_file, "rb"))

import re
matcher = re.compile(r"([\d\.\-]+) ± ([\d\.]+) \[\d+\]")

import pandas as pd
import matplotlib.pyplot as plt

entries = []

for table in [data]:
    keys = list(table.index.names)

    for index, line in table.iterrows():
        m = matcher.match(line["entropy"]["agg"])
        assert m, str(line)

        entry = dict(zip(keys, index))
        entry["entropy"] = float(m.group(1))
        entry["entropy_std"] = float(m.group(2))
    
        entries.append(entry)

df = pd.DataFrame(entries)

df["dataset2.num_last"] = df["dataset2.num_last"].astype('int')
df["opt.lr_z"] = df["opt.lr_z"].astype("float")
df["opt.lr_y_multi_on_z"] = df["opt.lr_y_multi_on_z"].astype("int")

df = df.sort_values(by=["dataset2.num_last", "opt.lr_z", "opt.lr_y_multi_on_z"])

plt.figure(figsize=(12,3))

for i, num_last in enumerate([1, 3, 5, 10]):
    df1 = df[ (df["dataset2.num_last"] == num_last) ]
    
    plt.subplot(1, 4, i + 1)
    for lr_z, tbl in df1.groupby(["opt.lr_z"]): 
        if lr_z == 5:
            continue
        plt.errorbar(tbl["opt.lr_y_multi_on_z"], tbl["entropy"], tbl["entropy_std"], label=fr"$\eta_Z = {lr_z}$")

    plt.axvline(1, color='k', linestyle='--', linewidth=0.5)
    plt.title(f"#last/next tokens = {num_last}/{num_last * 2}")
    plt.xlabel(r"$\eta_Y / \eta_Z$")

    if i == 0:
        plt.legend()
        plt.ylabel(r"Average Entropy of $\mathbf{c}_n$")
    
# plt.show()
plt.tight_layout()
plt.savefig("syn_med_entropy.pdf")
