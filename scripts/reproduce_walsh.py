"""Reproduce the historical Walsh comparison, including its fixed inner C grid."""
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from reproduce_ridge import grouped

ROOT=Path(__file__).resolve().parents[1]

def model(columns,c):
    return Pipeline([
        ('select',ColumnTransformer([('numeric',Pipeline([
            ('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),columns)],remainder='drop')),
        ('model',LogisticRegression(C=c,penalty='elasticnet',l1_ratio=.5,solver='saga',
                                   class_weight='balanced',max_iter=5000,random_state=42))])

def choose_c(frame,y,g,columns):
    candidates=(.01,.1,1.,10.)
    n=min(4,len(np.unique(g)))
    if n<3: return 1.,[]
    splits=list(GroupKFold(n_splits=n).split(frame,y,g))
    assert all(not set(g[a])&set(g[b]) for a,b in splits)
    scores=[]; eligible=[]
    for c in candidates:
        values=[]
        for train,valid in splits:
            if len(np.unique(y[train]))<2 or len(np.unique(y[valid]))<2: continue
            fit=model(columns,c).fit(frame.iloc[train],y[train])
            values.append(average_precision_score(y[valid],fit.predict_proba(frame.iloc[valid])[:,1]))
        scores.append(float(np.mean(values)) if values else -np.inf)
        eligible.append(len(values))
    return candidates[int(np.argmax(scores))],eligible

def main():
    # Report dropped columns explicitly below instead of repeating long sklearn warnings.
    warnings.filterwarnings('ignore',message='Skipping features without any observed values:',category=UserWarning)
    data=ROOT/'data'
    features=pd.read_csv(data/'walsh_host_features.csv',dtype={'host_id':str,'biosample':str})
    observations=pd.read_csv(data/'walsh_observations_st_bin.csv',dtype={'host_id':str,'biosample':str,'cc':str})
    assert not {'outcome','label','susceptible_primary','susceptible_strict','mean_final_od600'}&set(features.columns)
    assert features.qc_pass.dtype==bool
    features=features[features.qc_pass].copy()
    columns=['genome_length','gc_fraction','contig_count','n50']+sorted(c for c in features if c.startswith(('gene_','hsdS_','defensefinder_','crispr_','genomad_')))
    task=observations.merge(features,on=['host_id','biosample'],how='inner',validate='one_to_one')
    saved=pd.read_csv(data/'walsh_split_predictions.csv',dtype={'host_id':str,'biosample':str,'cc':str}).set_index('biosample').loc[task.biosample]
    y=task.susceptible_primary.to_numpy(int); g=task.cc.to_numpy(str)
    assert len(task)==45 and len(columns)==146 and y.sum()==29
    entirely_missing=[c for c in columns if task[c].isna().all()]
    available=[c for c in columns if c not in entirely_missing]
    np.testing.assert_array_equal(saved.label,y)
    np.testing.assert_array_equal(saved.host_id,task.host_id)
    np.testing.assert_array_equal(saved.cc,g)
    designs={'random_stratified':list(StratifiedKFold(5,shuffle=True,random_state=20260823).split(y,y)),
             'complete_cc':list(grouped(g))}
    results={}; outputs=task[['host_id','biosample','cc']].copy()
    for design,splits in designs.items():
        p=np.full(len(y),np.nan); counts=np.zeros(len(y),int); details=[]
        for fold,(train,test) in enumerate(splits):
            overlap=len(set(g[train])&set(g[test]))
            if design=='complete_cc': assert overlap==0
            assert not set(train)&set(test)
            c,eligible=choose_c(task.iloc[train],y[train],g[train],columns)
            fit=model(columns,c).fit(task.iloc[train],y[train])
            p[test]=fit.predict_proba(task.iloc[test])[:,1]; counts[test]+=1
            details.append({'fold':fold,'C':c,'eligible_inner_folds_by_C':eligible,
                            'train_n':len(train),'test_n':len(test),'cc_overlap':overlap,
                            'effective_feature_count':int(fit.named_steps['select'].transform(task.iloc[test]).shape[1]),
                            'solver_n_iter':int(fit.named_steps['model'].n_iter_.max())})
            print(design,fold,'complete',flush=True)
        assert np.all(counts==1) and np.isfinite(p).all()
        discrepancy=float(np.max(np.abs(saved['probability_'+design].to_numpy()-p)))
        assert discrepancy<1e-10,discrepancy
        results[design]={'max_prediction_error':discrepancy,'roc_auc':roc_auc_score(y,p),
                          'average_precision':average_precision_score(y,p),'folds':details}
        outputs['probability_'+design]=p
    out=ROOT/'results/walsh_reproduction'; out.mkdir(parents=True,exist_ok=True)
    outputs.to_csv(out/'reproduced_predictions.csv',index=False)
    inputs=['walsh_host_features.csv','walsh_observations_st_bin.csv','walsh_split_predictions.csv','WALSH_REPRODUCTION_PROTOCOL.json']
    result={'status':'COMPLETE','input_sha256':{f:hashlib.sha256((data/f).read_bytes()).hexdigest() for f in inputs},
            'input_text_lf_sha256':{f:hashlib.sha256((data/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in inputs},
            'n_hosts':len(task),'n_candidate_features':len(columns),
            'entirely_missing_columns':entirely_missing,'available_columns':available,'designs':results,
            'scope':'Exact historical refit, not new validation; different assay cohort with close cross-cohort genomic relatives.'}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'COMPLETE','n_hosts':len(task),'available_columns':available,
                      'max_prediction_errors':{k:v['max_prediction_error'] for k,v in results.items()}},indent=2))

if __name__=='__main__': main()
