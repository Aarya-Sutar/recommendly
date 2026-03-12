import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

def load_data(path):
    df = pd.read_csv(path)
    df = df[['visitorid','itemid','event','timestamp']]
    return df

def preprocess(df):
    event_weights = {
        'view':1.0,
        'addtocart':3.0,
        'transaction':5.0
    }
    df['weight'] = df['event'].map(event_weights)
    df = df.dropna()
    return df

def train_test_split_time(df):
    df = df.sort_values('timestamp')
    cutoff = int(len(df)*0.8)
    train = df.iloc[:cutoff]
    test = df.iloc[cutoff:]
    return train,test

def create_sparse_matrix(df):
    user_ids = df['visitorid'].astype("category").cat.codes
    item_ids = df['itemid'].astype("category").cat.codes

    matrix = csr_matrix(
        (df['weight'], (user_ids,item_ids))
    )

    return matrix,user_ids,item_ids

def recall_at_k(model,train_matrix,test_df,K=10):
    recall_scores = []
    for user in test_df['visitorid'].unique():
        try:
            user_index = user
            recommended = model.recommend(user_index,train_matrix[user_index],N=K)
            recommended_items = [r[0] for r in recommended]

            true_items = test_df[test_df['visitorid']==user]['itemid'].values
            hits = len(set(recommended_items) & set(true_items))
            recall = hits/len(true_items)
            recall_scores.append(recall)
        except:
            continue
    return np.mean(recall_scores)

def ndcg_at_k(model,train_matrix,test_df,K=10):
    def dcg(scores):
        return sum(
            score/np.log2(idx+2)
            for idx,score in enumerate(scores)
        )
    ndcg_scores = []
    for user in test_df['visitorid'].unique():
        try:
            recommended = model.recommend(user,train_matrix[user],N=K)
            recommended_items = [r[0] for r in recommended]

            true_items = test_df[test_df['visitorid']==user]['itemid'].values
            relevance = [1 if item in true_items else 0 for item in recommended_items]
            dcg_score = dcg(relevance)
            ideal_dcg = dcg(sorted(relevance,reverse=True))
            if ideal_dcg ==0:
                continue
            ndcg_scores.append(dcg_score/ideal_dcg)
        except:
            continue
    return np.mean(ndcg_scores)

def main():
    df = load_data("data/events.csv")
    df = preprocess(df)
    train_df,test_df = train_test_split_time(df)
    train_matrix,_,_= create_sparse_matrix(train_df)
    model = AlternatingLeastSquares(
        factors=64,
        regularization=0.1,
        iterations=20
    )
    model.fit(train_matrix)
    recall = recall_at_k(model,train_matrix,test_df)
    ndcg = ndcg_at_k(model,train_matrix,test_df)

    print(f"Recall@10: {recall}")
    print(f"NDCG@10: {ndcg}")

    with open("app/models/als_model.pkl","wb") as f:
        pickle.dump(model,f)

if __name__ == "__main__":
    main()
 