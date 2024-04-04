import pandas as pd
import json

class MeanStdAggFunc:
    def __init__(self, precision=2, use_latex=False):
        self.precision = precision
        self.use_latex = use_latex

    def agg(self, series):
        mean_val = series.mean()
        std_val = series.std()
        nona_cnt = series.count()
        if self.use_latex:
            return fr"${mean_val:.{self.precision}f}\pm {std_val:.{self.precision}f}$"
        else:
            return f"{mean_val:.{self.precision}f} ± {std_val:.{self.precision}f} [{nona_cnt}]"


with open("hier_align.json", "r") as f:
    results = json.load(f)

df = pd.DataFrame(results)

def agg_stats(key_stats):
    # Add aggregation function for each key_stats
    agg_obj = MeanStdAggFunc(precision=2, use_latex=True)

    aggs = { col: [ agg_obj.agg ] for col in key_stats }
    # aggs.update({ col + "_len": [ agg_obj.agg ] for col in key_stats })

    return aggs

df = pd.DataFrame(results)

import pdb
pdb.set_trace()

df = df[df["opt.wd"] == "0.0001"]
df = df[df["opt.lr"] == "0.0001"]
df = df[df["model.d"] == "1024"]

mapping = {
    "[2,2,2]" : 2,
    "[3,3,3]" : 3
}

mapping2 = {
    "[10,20,null]" : (10,20),
    "[20,30,null]" : (20,30)
}

df["gen.num_combinations"] = df["gen.num_combinations"].apply(lambda x : mapping[x])
df["gen.num_tokens"] = df["gen.num_tokens"].apply(lambda x : mapping2[x])

grouped = df.groupby(["num_class", "gen.num_combinations", "gen.num_tokens"]).agg(agg_stats(["score0", "score1"]))

latex = grouped.to_latex()

# from tabulate import tabulate

# latex_table_custom = tabulate(grouped, tablefmt='latex', headers='keys')
# print(latex_table_custom)

print(latex.replace(r"\textbackslash ", "\\").replace("\\$", "$"))
