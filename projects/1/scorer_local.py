#!/opt/conda/envs/dsenv/bin/python

#
# This is a Log Loss scorer
#

import sys
import os
import logging
import pandas as pd
from sklearn.metrics import log_loss
#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#
# Read true values
#
try:
    true_path, pred_path = sys.argv[1:]
except:
    logging.critical("Parameters: true_path (local) and pred_path (url or local)")
    sys.exit(1)

logging.info(f"TRUE PATH {true_path}")
logging.info(f"PRED PATH {pred_path}")


#open true path
df_true = pd.read_csv(true_path, sep='\t', header=None, index_col=False, names=["id", "true"])

#open pred_path
df_pred = pd.read_table(pred_path, sep='\t', header=None, index_col=False, names=["id", "pred"])

logging.info(df_pred.head())
logging.info(df_pred.columns)

df_pred_id = df_pred['id']

df_true = df_true.loc[df_pred_id, :]

len_true = len(df_true)
len_pred = len(df_pred)

logging.info(f"TRUE RECORDS {len_true}")
logging.info(f"PRED RECORDS {len_pred}")

logging.info(f"TRUE COLS\n{df_true.columns}")
logging.info(f"PRED COLS\n{df_pred.columns}")

assert len_true == len_pred, f"Number of records differ in true and predicted sets"

# df = df_true.join(df_pred, on='id')
# len_df = len(df)
# assert len_true == len_df, f"Combined true and pred has different number of records: {len_df}"

score = log_loss(df_true['true'], df_pred['pred'])

print(score)

sys.exit(0)

