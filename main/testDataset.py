import pandas as pd
import pickle

df = pd.read_csv('../data/TrainLabel.csv')

with open("../table/table1.txt", "rb") as fp:
    table1 = pickle.load(fp)
print(len(table1))