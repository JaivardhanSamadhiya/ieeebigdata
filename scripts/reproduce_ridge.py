"""Refit the fixed quantitative ridge pipeline from the released sparse matrix.

No model selection is performed. Family construction was global and outcome-blind;
only prevalence filtering and scaling are fold-local. This is not reconstruction
from raw sequencing reads. Run with Python 3.11, numpy 1.26.4, scipy 1.13.1,
pandas 2.2.2, scikit-learn 1.4.2 (versions recorded in run metadata).
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.stats import spearmanr
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
PHAGES=['p11','p0006','p0017','p0017S','p002y','p003p','p0040','pyo']

def grouped(groups):
    unique, counts=np.unique(groups.astype(str),return_counts=True)
    order=sorted(zip(unique,counts),key=lambda z:(-z[1],hashlib.sha256(z[0].encode()).hexdigest()))
    loads=[0]*5; assignment={}
    for group,n in order:
        fold=min(range(5),key=lambda j:(loads[j],j))
        assignment[group]=fold; loads[fold]+=int(n)
    for fold in range(5):
        test=np.flatnonzero([assignment[g]==fold for g in groups])
        yield np.setdiff1d(np.arange(len(groups)),test),test

def predict(x,y,groups,splits):
    pred=np.full(y.shape,np.nan); null=np.full(y.shape,np.nan); sizes=[]
    for train,test in splits:
        prevalence=np.asarray(x[train].mean(axis=0)).ravel()
        keep=(prevalence>=.02)&(prevalence<=.98)
        scaler=StandardScaler(with_mean=False)
        xt=scaler.fit_transform(x[train][:,keep]); xv=scaler.transform(x[test][:,keep])
        for j in range(y.shape[1]):
            model=Ridge(alpha=10.,solver='lsqr').fit(xt,y[train,j])
            pred[test,j]=model.predict(xv)
        null[test]=y[train].mean(axis=0)
        sizes.append({'n_train':len(train),'n_test':len(test),'n_features':int(keep.sum()),
                      'cc_overlap':len(set(groups[train])&set(groups[test]))})
    assert np.isfinite(pred).all()
    return pred,null,sizes

def metrics(y,p):
    return {'spearman':float(np.mean([spearmanr(y[:,j],p[:,j]).statistic for j in range(8)])),
            'r2':float(np.mean([r2_score(y[:,j],p[:,j]) for j in range(8)])),
            'mae':float(mean_absolute_error(y,p))}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeat-seeds',action='store_true',help='Run all 20 predeclared seeds 2026091900..2026091919; do not select a favorable split.')
    args=parser.parse_args()
    x=sparse.load_npz(DATA/'pangenome_presence_absence.npz').tocsr().astype(float)
    hosts=pd.read_csv(DATA/'pangenome_hosts.csv',dtype={'biosample':str})
    observed=pd.read_csv(DATA/'evaluation_design_predictions.csv',dtype={'biosample':str,'cc':str})
    source=observed[observed.design=='random_kfold']
    assert not source.duplicated(['biosample','phage']).any()
    y=source.pivot(index='biosample',columns='phage',values='observed_od600').loc[hosts.biosample,PHAGES].to_numpy()
    g=source[['biosample','cc']].drop_duplicates().set_index('biosample').loc[hosts.biosample,'cc'].to_numpy()
    assert x.shape[0]==len(hosts)==217 and y.shape==(217,8)
    strata=np.char.add(np.char.add(np.sum(y<.4,axis=1).astype(str),'_'),np.sum(y>=.7,axis=1).astype(str))
    counts=pd.Series(strata).value_counts()
    strata=np.array([s if counts[s]>=5 else 'rare' for s in strata])
    splits={
        'random_kfold':list(KFold(5,shuffle=True,random_state=20260821).split(y)),
        'outcome_stratified_kfold':list(StratifiedKFold(5,shuffle=True,random_state=20260821).split(y,strata)),
        'complete_cc_holdout':list(grouped(g))}
    result={'versions':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
            'matrix_shape':list(x.shape),'designs':{}}
    for design,folds in splits.items():
        p,null,sizes=predict(x,y,g,folds)
        reference=observed[observed.design==design].pivot(index='biosample',columns='phage',values='predicted_od600').loc[hosts.biosample,PHAGES].to_numpy()
        error=float(np.max(np.abs(reference-p)))
        if error>1e-6:
            raise AssertionError(f'{design}: stored/refit discrepancy {error}; inspect numerical environment')
        if design=='complete_cc_holdout': assert all(s['cc_overlap']==0 for s in sizes)
        result['designs'][design]={'ridge':metrics(y,p),'training_mean_baseline':metrics(y,null),'folds':sizes,'max_prediction_discrepancy':error}
        print(design,result['designs'][design]['ridge'],flush=True)
    if args.repeat_seeds:
        result['retrospective_random_seed_sensitivity']=[]
        for seed in range(2026091900,2026091920):
            p,_,_=predict(x,y,g,list(KFold(5,shuffle=True,random_state=seed).split(y)))
            row={'seed':seed,**metrics(y,p)}
            result['retrospective_random_seed_sensitivity'].append(row)
            print(row,flush=True)
    out=ROOT/'results'; out.mkdir(exist_ok=True)
    (out/'ridge_reproduction.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
