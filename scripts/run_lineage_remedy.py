"""Exploratory fixed within-lineage ridge remedy and prespecified ablations."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT
from reproduce_ridge import PHAGES, grouped, metrics
OUT=ROOT/'results/lineage_remedy'
MODELS=['pooled_ridge','balanced_ridge','within_lineage_ridge','isolate_mean','lineage_mean']

def fit_fold(x,y,g,train,test):
    assert not set(g[train]) & set(g[test])
    prev=np.asarray(x[train].mean(axis=0)).ravel()
    keep=(prev>=.02)&(prev<=.98)
    scaler=StandardScaler(with_mean=False)
    xt=scaler.fit_transform(x[train][:,keep]).toarray()
    xv=scaler.transform(x[test][:,keep]).toarray()
    yt=y[train]; gt=g[train]; unique=np.unique(gt)
    w=np.array([len(train)/(len(unique)*np.sum(gt==cc)) for cc in gt])
    assert np.isclose(w.sum(),len(train))
    xm=np.array([xt[gt==cc].mean(axis=0) for cc in unique])
    ym=np.array([yt[gt==cc].mean(axis=0) for cc in unique])
    lookup={cc:i for i,cc in enumerate(unique)}
    row=np.array([lookup[cc] for cc in gt])
    xc=xt-xm[row]; yc=yt-ym[row]
    results={m:np.zeros((len(test),8)) for m in MODELS}
    max_invariance_error=0.
    for j in range(8):
        for name,weights in [('pooled_ridge',None),('balanced_ridge',w)]:
            # Keep the original sparse implementation for exact baseline reproduction.
            xb=sparse.csr_matrix(xt) if name=='pooled_ridge' else xt
            vb=sparse.csr_matrix(xv) if name=='pooled_ridge' else xv
            fit=Ridge(alpha=10.,solver='lsqr').fit(xb,yt[:,j],sample_weight=weights)
            results[name][:,j]=fit.predict(vb)
        fit=Ridge(alpha=10.,solver='lsqr',fit_intercept=False,tol=1e-10).fit(xc,yc[:,j],sample_weight=w)
        results['within_lineage_ridge'][:,j]=ym[:,j].mean()+(xv-xm.mean(axis=0))@fit.coef_
        # Verify the defining within-lineage invariance on one response per fold.
        if j==0:
            altered=yt[:,j]+(row+1)*.37
            centered=altered-np.array([altered[gt==cc].mean() for cc in unique])[row]
            check=Ridge(alpha=10.,solver='lsqr',fit_intercept=False,tol=1e-10).fit(xc,centered,sample_weight=w)
            max_invariance_error=float(np.max(np.abs(fit.coef_-check.coef_)))
            assert max_invariance_error<1e-8, max_invariance_error
    results['isolate_mean'][:]=yt.mean(axis=0)
    results['lineage_mean'][:]=ym.mean(axis=0)
    return results,{'n_train':len(train),'n_test':len(test),'n_train_cc':len(unique),
                    'n_features':int(keep.sum()),'invariance_max_coefficient_error':max_invariance_error}

def main():
    protocol=ROOT/'data/LINEAGE_REMEDY_PROTOCOL.json'
    OUT.mkdir(parents=True,exist_ok=True)
    assert json.loads(protocol.read_text())['status']=='RETROSPECTIVE_REMEDY_PROTOCOL_BEFORE_NEW_FITS'
    d=PUBLIC/'data'
    hosts=pd.read_csv(d/'pangenome_hosts.csv')
    original=pd.read_csv(d/'evaluation_design_predictions.csv',dtype={'cc':str})
    source=original[original.design=='complete_cc_holdout']
    y=source.pivot(index='biosample',columns='phage',values='observed_od600').loc[hosts.biosample,PHAGES].to_numpy()
    metadata=source[['biosample','host_id','cc']].drop_duplicates().set_index('biosample').loc[hosts.biosample]
    g=metadata.cc.to_numpy()
    x=sparse.load_npz(d/'pangenome_presence_absence.npz').tocsr().astype(float)
    designs={'complete_cc_fivefold':list(grouped(g)),
             'leave_one_cc_out':[(np.flatnonzero(g!=cc),np.flatnonzero(g==cc)) for cc in np.unique(g)]}
    summary={}; output=[]
    rng=np.random.default_rng(20260919)
    unique=np.unique(g)
    bootstrap_indices=rng.integers(0,len(unique),(2000,len(unique)))
    for design,folds in designs.items():
        pred={m:np.full(y.shape,np.nan) for m in MODELS}; details=[]
        for f,(train,test) in enumerate(folds):
            p,detail=fit_fold(x,y,g,train,test)
            for model in MODELS: pred[model][test]=p[model]
            details.append(detail)
            print(design,'fold',f,'completed',flush=True)
        assert all(np.isfinite(p).all() for p in pred.values())
        if design=='complete_cc_fivefold':
            ref=source.pivot(index='biosample',columns='phage',values='predicted_od600').loc[hosts.biosample,PHAGES].to_numpy()
            discrepancy=float(np.max(np.abs(pred['pooled_ridge']-ref)))
            assert discrepancy<1e-6,discrepancy
        by_cc={m:np.array([np.abs(y[g==cc]-pred[m][g==cc]).mean() for cc in unique]) for m in MODELS}
        scores={m:{**metrics(y,pred[m]),'equal_cc_mae':float(by_cc[m].mean()),
                   'per_cc_mae':dict(zip(unique,by_cc[m].tolist()))} for m in MODELS}
        contrasts={}
        for baseline in ['pooled_ridge','balanced_ridge','isolate_mean','lineage_mean']:
            delta=by_cc[baseline]-by_cc['within_lineage_ridge']
            boot=delta[bootstrap_indices].mean(axis=1)
            contrasts[baseline]={'baseline_minus_within_mae':float(delta.mean()),
                                 'conditional_cc_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
                                 'lineages_improved':int((delta>0).sum())}
        summary[design]={'models':scores,'contrasts':contrasts,'folds':details}
        for m in MODELS:
            for i in range(len(y)):
                for j,phage in enumerate(PHAGES):
                    output.append({'design':design,'model':m,'biosample':hosts.iloc[i].biosample,
                                   'cc':g[i],'phage':phage,'observed':float(y[i,j]),'predicted':float(pred[m][i,j])})
        print(json.dumps({design:{m:scores[m]['equal_cc_mae'] for m in MODELS}}),flush=True)
    pd.DataFrame(output).to_csv(OUT/'predictions.csv',index=False)
    result={'status':'COMPLETE','protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
            'interpretation':'Retrospective exploratory remedy comparison; no independent clinical validation.', 'designs':summary}
    (OUT/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',flush=True)

if __name__=='__main__': main()
