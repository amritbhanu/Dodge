from __future__ import print_function, division
import os
import sys
import pandas as pd
from pdb import set_trace
from glob import glob
from tabulate import tabulate


def about():
    rows = []
    for dir in os.listdir(os.getcwd()):
        formatted_path = os.path.join(os.getcwd(), dir)
        if os.path.isdir(formatted_path):
            print(dir)
            head = ["Dataset", "Days", "Samples", " Smelly (%)", "# metrics", "Nature"]
            files = glob(os.path.join(os.path.abspath(dir), "*.csv"))
            nature = "Class" if dir == "DataClass" or dir == "GodClass" else "Method"
            for file in files:
                name = file.split("/")[-1].split(".csv")[0]
                name = name.split("-")[0]
                dframe = pd.read_csv(file)
                N = len(dframe)
                n_metrics = len(dframe.columns)
                smells = sum([1 if s > 0 else 0 for s in dframe["timeOpen"].values])
                p_smells = round(100 * smells / N, 0)
                rows.append([name, dir, N, "{} ({})".format(smells, p_smells), n_metrics, nature])
            # set_trace()
    stats = pd.DataFrame(rows, columns=head)
    stats.set_index("Dataset", inplace=False)
    stats.sort_index()
    stats.to_csv(os.path.abspath(os.path.join(".", "all_data.csv")), index=False)
    print(tabulate(stats, headers=head, showindex="never"), end="\n\n")
    set_trace()


if __name__ == "__main__":
    about()
