import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')
parser.add_argument('--hpc', action='store_true', help='Whether running on HPC or not')
args = parser.parse_args()

root = '/home/svu/e0407728/My_FYP/' if args.hpc else '/workspaces/'


def preprocess(data_name): #original data: user_id, item_id, timestamp, label, comma-separated-features
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f): #index, line per line reading
      e = line.strip().split(',')
      u = int(e[0])
      i = int(e[1])

      ts = float(e[2])
      label = float(e[3])  # int(e[3])

      feat = np.array([float(x) for x in e[4:]]) #all other features

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)

      feat_l.append(feat)
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite: #check bipartiteness
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique())) 
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1 #upperbound exclusive
    new_i = df.i + upper_u #because the destination node id also starts from 0 like srouce, to make it less confusing.

    new_df.i = new_i
    new_df.u += 1 #why all + 1?
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df


def run(data_name, bipartite=True):
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = root + 'data/{}.csv'.format(data_name)
  OUT_DF = root +'tgn/data/ml_{}.csv'.format(data_name)
  OUT_FEAT = root + 'tgn/data/ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = root + 'tgn/data/ml_{}_node.npy'.format(data_name)

  df, feat = preprocess(PATH) #separation of graph structure and node features
  new_df = reindex(df, bipartite)

  empty = np.zeros(feat.shape[1])[np.newaxis, :] #make 0 for all features, according to the number of features, in a row
  feat = np.vstack([empty, feat]) #stack all 0 in the top, all features below
  #why? now have 10 features but 9 rows

  max_idx = max(new_df.u.max(), new_df.i.max())
  rand_feat = np.zeros((max_idx + 1, 172)) #max_idx rows, 172 cols of 0

  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, rand_feat)

run(args.data, bipartite=args.bipartite)