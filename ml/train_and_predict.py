# train_and_predict.py
import argparse, warnings, os
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sqlalchemy import create_engine, text
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def find_tables(engine):
    with engine.connect() as conn:
        res = conn.execute(text("SHOW TABLES;"))
        return [r[0] for r in res]

def load_table(engine, table):
    return pd.read_sql_table(table, con=engine)

def guess_label_column(df):
    candidates = ['is_fraud','isfraud','fraud','class','label','status','flag','predicted']
    for c in candidates:
        for col in df.columns:
            if col.lower()==c:
                return col
    return None

def create_labels_from_fraud_table(df_cc, fraud_df):
    cc_cols = [c.lower() for c in df_cc.columns]
    fraud_cols = [c.lower() for c in fraud_df.columns]
    common = set(cc_cols) & set(fraud_cols)
    if common:
        c_cc = [c for c in df_cc.columns if c.lower() in common][0]
        c_f = [c for c in fraud_df.columns if c.lower() in common][0]
        keys = set(fraud_df[c_f].astype(str).str.strip().dropna().unique())
        return df_cc[c_cc].astype(str).str.strip().isin(keys).astype(int)
    for c1 in df_cc.columns:
        vals1 = set(df_cc[c1].astype(str).str.strip().dropna().unique()[:2000])
        if not vals1: continue
        for c2 in fraud_df.columns:
            vals2 = set(fraud_df[c2].astype(str).str.strip().dropna().unique()[:2000])
            if not vals2: continue
            inter = len(vals1 & vals2)
            if inter>0 and inter/len(vals1)>0.01:
                return df_cc[c1].astype(str).str.strip().isin(vals2).astype(int)
    return None

def preprocess_X(df):
    drop_like = ['id','date','time','timestamp','name','email','mobile','address','image','photo','password','username']
    X = df.copy()
    for col in list(X.columns):
        lc = col.lower()
        if any(d in lc for d in drop_like):
            X.drop(columns=col, inplace=True, errors=True)
    numeric_cols = []
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            numeric_cols.append(col)
        except:
            pass
    Xnum = X[numeric_cols].fillna(0.0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xnum)
    return Xs, numeric_cols, scaler

def main(args):
    engine = create_engine(f"mysql+pymysql://{args.user}:{args.password}@{args.host}:{args.port}/{args.database}")
    print('Connected to DB')
    tables = find_tables(engine)
    print('Tables found:', tables)
    cc_candidates = [t for t in tables if 'credit' in t.lower() or ('card' in t.lower() and 'request' not in t.lower())]
    fraud_candidates = [t for t in tables if 'fraud' in t.lower()]
    if not cc_candidates:
        print('No credit-like table found. Exiting')
        return
    credit_table = cc_candidates[0]
    df_cc = load_table(engine, credit_table)
    print('Loaded table', credit_table, 'shape', df_cc.shape)
    label_col = guess_label_column(df_cc)
    y = None
    if label_col:
        print('Found label column in credit table:', label_col)
        y = df_cc[label_col].astype(int)
    else:
        for ft in fraud_candidates:
            df_f = load_table(engine, ft)
            lbl = create_labels_from_fraud_table(df_cc, df_f)
            if lbl is not None:
                y = lbl
                print('Labels created from', ft)
                break
    if y is None:
        print('No labels found; creating dummy zeros')
        y = pd.Series(np.zeros(len(df_cc), dtype=int))
    Xs, numeric_cols, scaler = preprocess_X(df_cc)
    if Xs.shape[1]==0:
        print('No numeric features detected. Columns:')
        print(df_cc.columns.tolist())
        return
    try:
        X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)
    except:
        X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=42)
    try:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print('Applied SMOTE')
    except Exception as e:
        print('SMOTE failed:', e)
    ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, learning_rate=0.5, random_state=42)
    ada.fit(X_train, y_train)
    print('AdaBoost trained')
    y_pred = ada.predict(X_test)
    print('Classification report:\n', classification_report(y_test, y_pred, digits=4))
    try:
        y_proba = ada.predict_proba(X_test)[:,1]
        print('ROC AUC:', roc_auc_score(y_test, y_proba))
    except:
        pass
    full_preds = ada.predict(Xs)
    try:
        full_score = ada.predict_proba(Xs)[:,1]
    except:
        full_score = np.zeros(len(full_preds))
    pred_df = pd.DataFrame({'prediction': full_preds.astype(int), 'score': full_score})
    id_col = None
    for col in df_cc.columns:
        if col.lower() in ('id','txn_id','trans_id','tid','cardid','card_id'):
            id_col = col
            break
    if id_col:
        pred_df[id_col] = df_cc[id_col].values
    else:
        pred_df['row_idx'] = df_cc.index.values
    with engine.begin() as conn:
        conn.execute(text('DROP TABLE IF EXISTS fraud_predictions;'))
        pred_df.to_sql('fraud_predictions', con=engine, index=False)
    joblib.dump({'model': ada, 'scaler': scaler, 'numeric_cols': numeric_cols}, 'ada_model.joblib')
    print('Wrote fraud_predictions table and saved model ada_model.joblib')

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='db')
    parser.add_argument('--port', default=3306, type=int)
    parser.add_argument('--user', default='root')
    parser.add_argument('--password', default='root')
    parser.add_argument('--database', default='cc_fraud')
    args = parser.parse_args()
    main(args)
