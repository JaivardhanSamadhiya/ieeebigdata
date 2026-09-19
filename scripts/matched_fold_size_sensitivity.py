"""Retrospective random-fold control matching complete-CC fold sizes exactly."""
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import sparse
from reproduce_ridge import ROOT, DATA, PHAGES, grouped, predict, metrics

def main():
    protocol=DATA/'MATCHED_FOLD_SIZE_PROTOCOL.json'
    assert json.loads(protocol.read_text())['status']=='RETROSPECTIVE_SENSITIVITY_PLAN_BEFORE_MATCHED_SIZE_REFITS'
    hosts=pd.read_csv(DATA/'pangenome_hosts.csv')
    source=pd.read_csv(DATA/'evaluation_design_predictions.csv',dtype={'cc':str})
    source=source[source.design=='random_kfold']
    y=source.pivot(index='biosample',columns='phage',values='observed_od600').loc[hosts.biosample,PHAGES].to_numpy()
    g=source[['biosample','cc']].drop_duplicates().set_index('biosample').loc[hosts.biosample,'cc'].to_numpy()
    x=sparse.load_npz(DATA/'pangenome_presence_absence.npz').tocsr().astype(float)
    sizes=[len(test) for _,test in grouped(g)]
    assert sizes==[88,50,27,26,26]
    rows=[]
    for seed in range(2026092000,2026092020):
        shuffled=np.random.default_rng(seed).permutation(len(g))
        tests=np.split(shuffled,np.cumsum(sizes)[:-1])
        folds=[(np.setdiff1d(np.arange(len(g)),test),test) for test in tests]
        p,_,detail=predict(x,y,g,folds)
        row={'seed':seed,**metrics(y,p),'fold_cc_overlap':[d['cc_overlap'] for d in detail]}
        rows.append(row); print(row,flush=True)
    result={'status':'COMPLETE','protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
            'test_sizes':sizes,'all_seeds':rows,
            'ranges':{k:[min(r[k] for r in rows),max(r[k] for r in rows)] for k in ['spearman','r2','mae']}}
    out=ROOT/'results'; out.mkdir(exist_ok=True)
    (out/'matched_fold_size_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
