"""Fixed training-genome support diagnostic; no outcome-based gate tuning."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from reproduce_ridge import DATA, ROOT, PHAGES, grouped

def support_gate(similarity,train,test):
    training=similarity[np.ix_(train,train)].copy()
    np.fill_diagonal(training,-np.inf)
    nearest=training.max(axis=1)
    assert np.isfinite(nearest).all()
    threshold=float(np.quantile(nearest,.05,method='linear'))
    test_scores=similarity[np.ix_(test,train)].max(axis=1)
    return test_scores,threshold,test_scores>=threshold

def mean_or_none(values):
    return float(np.mean(values)) if len(values) else None

def main():
    protocol=DATA/'GENOME_SUPPORT_PROTOCOL.json'
    assert json.loads(protocol.read_text())['status']=='RETROSPECTIVE_DIAGNOSTIC_PROTOCOL_BEFORE_GATE_COMPUTATION'
    hosts=pd.read_csv(DATA/'pangenome_hosts.csv')
    observed=pd.read_csv(DATA/'evaluation_design_predictions.csv',dtype={'cc':str})
    source=observed[observed.design=='random_kfold']
    y=source.pivot(index='biosample',columns='phage',values='observed_od600').loc[hosts.biosample,PHAGES].to_numpy()
    meta=source[['biosample','host_id','cc']].drop_duplicates().set_index('biosample').loc[hosts.biosample]
    g=meta.cc.to_numpy()
    lookup={b:i for i,b in enumerate(hosts.biosample)}
    pairs=pd.read_csv(DATA/'moller_pairwise_genome_similarity.csv')
    assert len(pairs)==217*216//2 and not pairs.duplicated(['biosample_a','biosample_b']).any()
    sim=np.full((217,217),np.nan)
    for row in pairs.itertuples():
        a,b=lookup[row.biosample_a],lookup[row.biosample_b]
        sim[a,b]=sim[b,a]=row.sourmash_jaccard_k31_scaled1000
    np.fill_diagonal(sim,1.)
    assert np.isfinite(sim).all() and np.allclose(sim,sim.T)
    strata=np.char.add(np.char.add(np.sum(y<.4,axis=1).astype(str),'_'),np.sum(y>=.7,axis=1).astype(str))
    counts=pd.Series(strata).value_counts()
    strata=np.array([s if counts[s]>=5 else 'rare' for s in strata])
    designs={'random_kfold':list(KFold(5,shuffle=True,random_state=20260821).split(y)),
             'outcome_stratified_kfold':list(StratifiedKFold(5,shuffle=True,random_state=20260821).split(y,strata)),
             'complete_cc_holdout':list(grouped(g))}
    gates=[]; baselines={}
    for design,folds in designs.items():
        baselines[design]=np.full(y.shape,np.nan)
        for fold,(train,test) in enumerate(folds):
            scores,threshold,accepted=support_gate(sim,train,test)
            baselines[design][test]=y[train].mean(axis=0)
            for i,score,accept in zip(test,scores,accepted):
                gates.append({'design':design,'fold':fold,'biosample':hosts.iloc[i].biosample,
                              'cc':g[i],'score':float(score),'threshold':threshold,'accepted':bool(accept),
                              'cc_represented_in_training':bool(g[i] in set(g[train]))})
    out=ROOT/'results/genome_support'; out.mkdir(parents=True,exist_ok=True)
    gates=pd.DataFrame(gates)
    assert len(gates)==651 and not gates.duplicated(['design','biosample']).any()
    gate_file=out/'gates_before_error_scoring.csv'
    gates.to_csv(gate_file,index=False)
    gate_hash=hashlib.sha256(gate_file.read_bytes()).hexdigest()
    result={}
    per_cc=[]
    for design in designs:
        gate=gates[gates.design==design].set_index('biosample').loc[hosts.biosample]
        pred=observed[observed.design==design].pivot(index='biosample',columns='phage',values='predicted_od600').loc[hosts.biosample,PHAGES].to_numpy()
        error=np.abs(y-pred).mean(axis=1)
        null_error=np.abs(y-baselines[design]).mean(axis=1)
        keep=gate.accepted.to_numpy(bool)
        represented=gate.cc_represented_in_training.to_numpy(bool)
        result[design]={'n':217,'n_accepted':int(keep.sum()),'coverage':float(keep.mean()),
                        'all_host_mae':float(error.mean()),'accepted_mae':mean_or_none(error[keep]),
                        'rejected_mae':mean_or_none(error[~keep]),'accepted_training_mean_mae':mean_or_none(null_error[keep]),
                        'accepted_cc_represented_fraction':mean_or_none(represented[keep]),
                        'rejected_cc_represented_fraction':mean_or_none(represented[~keep]),
                        'threshold_range':[float(gate.threshold.min()),float(gate.threshold.max())]}
        for cc in np.unique(g):
            m=g==cc; a=m&keep
            per_cc.append({'design':design,'cc':cc,'n':int(m.sum()),'n_accepted':int(a.sum()),
                           'coverage':float(keep[m].mean()),'accepted_mae':mean_or_none(error[a])})
    pd.DataFrame(per_cc).to_csv(out/'per_lineage_coverage.csv',index=False)
    summary={'status':'COMPLETE','protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
             'gate_before_error_scoring_sha256':gate_hash,'designs':result,
             'interpretation':'Outcome-free genomic-support screening; no guarantee of accuracy, clinical safety, or transfer.'}
    (out/'RESULT.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
