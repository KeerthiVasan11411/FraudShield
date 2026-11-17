import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'fraud_intelligence'
OUTPUT_DIR = DATA_DIR / 'ablation_reports'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASIC_FEATURES = ['amount','card_present','is_international','transaction_hour','transaction_day','is_weekend','account_age_days']
PERSONALIZED_FEATURES = ['distance_from_home_km','device_change','transactions_last_1h','transactions_last_24h','failed_attempts_24h','customer_prior_transaction_count','customer_avg_amount_prior','customer_std_amount_prior','amount_deviation_ratio','unusual_transaction_time','new_merchant_for_customer','new_payee','has_time_history','has_merchant_history','has_payee_history']
ALL_FEATURES = BASIC_FEATURES + PERSONALIZED_FEATURES
EXPERIMENTS = {
    'All Features': ALL_FEATURES,
    'Without Amount': [f for f in ALL_FEATURES if f != 'amount'],
    'Without Location': [f for f in ALL_FEATURES if f != 'distance_from_home_km'],
    'Without Amount + Location': [f for f in ALL_FEATURES if f not in {'amount','distance_from_home_km'}],
    'Behavioral Only': PERSONALIZED_FEATURES.copy(),
}

plt.rcParams.update({'figure.facecolor':'#0B1120','axes.facecolor':'#151F32','axes.edgecolor':'#94A3B8','axes.labelcolor':'#F1F5F9','text.color':'#F1F5F9','xtick.color':'#94A3B8','ytick.color':'#94A3B8','savefig.facecolor':'#0B1120'})

def load_data():
    # Test data intentionally NOT read in this experiment.
    data = pd.read_csv(DATA_DIR / 'train_data.csv')
    required = set(ALL_FEATURES + ['Class','timestamp','transaction_id'])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if data['Class'].isna().any() or not data['Class'].isin([0,1]).all():
        raise ValueError('Class must contain only 0 and 1')
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='raise')
    data = data.sort_values(['timestamp','transaction_id'], kind='stable').reset_index(drop=True)
    boundary = data.loc[int(len(data)*0.75),'timestamp']
    train = data.loc[data['timestamp'] < boundary].copy()
    val = data.loc[data['timestamp'] >= boundary].copy()
    if train.empty or val.empty or train['Class'].nunique()!=2 or val['Class'].nunique()!=2:
        raise ValueError('Training and validation each need both classes')
    print(f'Training: {len(train):,} | Fraud rate: {train.Class.mean()*100:.2f}%')
    print(f'Validation: {len(val):,} | Fraud rate: {val.Class.mean()*100:.2f}%')
    print('Training ends:',train.timestamp.max())
    print('Validation begins:',val.timestamp.min())
    return train,val

def get_features(data, columns):
    X=data[columns].copy()
    for col in columns:
        X[col]=pd.to_numeric(X[col].replace({'True':1,'False':0,'true':1,'false':0}),errors='coerce')
    X=X.replace([np.inf,-np.inf],np.nan)
    return X,data['Class'].astype(int)

def best_threshold(y,prob):
    p,r,t=precision_recall_curve(y,prob)
    f1=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12)
    idx=int(np.argmax(f1))
    return float(t[idx])

def models(y):
    neg=int((y==0).sum()); pos=int((y==1).sum())
    return {
        'Random Forest':RandomForestClassifier(n_estimators=200,max_depth=12,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=-1),
        'XGBoost':XGBClassifier(n_estimators=250,max_depth=5,learning_rate=0.05,subsample=0.85,colsample_bytree=0.85,scale_pos_weight=neg/pos,objective='binary:logistic',eval_metric='logloss',random_state=42,n_jobs=-1)
    }

def plot_results(results):
    fig,ax=plt.subplots(figsize=(12,6))
    labels=list(EXPERIMENTS)
    x=np.arange(len(labels)); width=0.35
    for i,(model,color) in enumerate([('Random Forest','#06B6D4'),('XGBoost','#A78BFA')]):
        subset=results[results['model']==model].set_index('experiment').reindex(labels)
        ax.bar(x+(i-0.5)*width,subset['average_precision'].values,width,label=model,color=color)
    ax.set_xticks(x,labels,rotation=18,ha='right')
    ax.set_ylabel('Validation Average Precision')
    ax.set_title('FraudShield Feature Ablation — Validation AP',pad=16)
    ax.set_ylim(0,1.05)
    ax.grid(axis='y',alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR/'ablation_average_precision.png',dpi=180,bbox_inches='tight')
    plt.close(fig)

def run_ablation():
    print('='*65)
    print('FRAUDSHIELD - FEATURE ABLATION EXPERIMENTS')
    print('='*65)
    train,val=load_data()
    rows=[]
    for exp,features in EXPERIMENTS.items():
        print('\n'+'='*65+'\n'+exp+'\n'+'='*65)
        X_train,y_train=get_features(train,features)
        X_val,y_val=get_features(val,features)
        for model_name,estimator in models(y_train).items():
            pipeline=Pipeline([
                ('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                ('classifier',estimator)
            ])
            pipeline.fit(X_train,y_train)
            prob=pipeline.predict_proba(X_val)[:,1]
            threshold=best_threshold(y_val,prob)
            pred=(prob>=threshold).astype(int)
            row={'experiment':exp,'model':model_name,'features':len(features),'threshold':threshold,
                 'precision':precision_score(y_val,pred,zero_division=0),
                 'recall':recall_score(y_val,pred,zero_division=0),
                 'f1_score':f1_score(y_val,pred,zero_division=0),
                 'average_precision':average_precision_score(y_val,prob)}
            rows.append(row)
            print(f"{model_name:14} | AP: {row['average_precision']:.4f} | F1: {row['f1_score']:.4f} | Precision: {row['precision']:.4f} | Recall: {row['recall']:.4f} | Threshold: {threshold:.4f}")
    results=pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR/'ablation_comparison.csv',index=False)
    baseline=results[results.experiment=='All Features'].set_index('model')['average_precision']
    results['ap_change_vs_full']=results.apply(lambda r:r['average_precision']-baseline.loc[r['model']],axis=1)
    results.to_csv(OUTPUT_DIR/'ablation_comparison.csv',index=False)
    plot_results(results)
    print('\nFINAL COMPARISON (AP difference vs same-model full feature set)')
    print(results[['experiment','model','average_precision','ap_change_vs_full','f1_score']].to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    print('\nReports saved to:',OUTPUT_DIR)
    print('Note: Thresholds were optimized on validation for descriptive comparison; these are not unbiased test metrics.')

if __name__=='__main__':
    run_ablation()
