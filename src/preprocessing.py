import pandas as pd
import numpy as np
import torch

def load_and_prep_paysim(file_path):
    """Loads PaySim data and filters for high-fraud transaction types[cite: 67]."""
    df = pd.read_csv(file_path)
    # Focus on TRANSFER and CASH_OUT as they contain nearly every fraud case [cite: 67]
    df = df[df['type'].isin(['CASH_OUT', 'TRANSFER'])].copy()
    return pd.get_dummies(df, columns=['type'], drop_first=False)

def build_transaction_graph(df, device='cpu'):
    """Converts transaction data into a directed graph structure[cite: 71, 72]."""
    senders = df['nameOrig'].astype('category')
    receivers = df['nameDest'].astype('category')
    
    # Unified account index
    all_nodes = pd.Index(senders.cat.categories).union(pd.Index(receivers.cat.categories))
    node_map = pd.Series(np.arange(len(all_nodes)), index=all_nodes)
    
    src = node_map.loc[senders.astype(str)].to_numpy()
    dst = node_map.loc[receivers.astype(str)].to_numpy()
    
    # Directed edges from payer to recipient [cite: 71]
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long).to(device)
    return edge_index, src, dst

def create_hybrid_features(df, embeddings, src_idx, dst_idx):
    """Merges Node2Vec embeddings with tabular attributes[cite: 85]."""
    tab_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrg', 
                'oldbalanceDest', 'newbalanceDest'] + \
               [c for c in df.columns if c.startswith('type_')]
    
    X_tab = df[tab_cols].fillna(0).to_numpy()
    X_src = embeddings[src_idx]
    X_dst = embeddings[dst_idx]
    
    # Hybrid set: sender embedding + receiver embedding + tabular data [cite: 85]
    X = np.hstack([X_src, X_dst, X_tab]).astype(np.float32)
    y = df['isFraud'].astype(int).to_numpy()
    return X, y
