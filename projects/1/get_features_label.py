#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load

import pandas as pd
sys.path.append('.')
from model import fields

model = load("1.joblib")

read_opts = dict(sep="\t", names=fields, index_col=False, header=None)
df = pd.read_csv("/home/users/datasets/criteo/criteo_train500", **read_opts)

df[['id', 'label']].to_csv("criteo_train500-target.csv", sep="\t", header=False, index=False)
df.drop('label', axis=1).to_csv("criteo_train500-features.csv", sep="\t", header=False, index=False)

print(df)