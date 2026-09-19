"""Retrospective influence checks on fixed predictions; never fits a model."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/hostbio_sensitivity_20260919'
M = ROOT / 'data/hostbio_final_mechanism'
H = ROOT / 'data/hostbio_submission_hardening'
if not M.exists():
    # The public scientific package uses a flat input directory.
    M = H = ROOT / 'data'
    OUT = ROOT / 'data/lineage_sensitivity'

def digest(p):
    # Text inputs: canonical LF newlines make provenance portable across Git checkouts.
    return hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()

def quantitative(frame):
    rows = []
    for (design, phage), block in frame.groupby(['design', 'phage']):
        y, p = block.observed_od600, block.predicted_od600
        rows.append(dict(design=design, phage=phage, spearman=float(spearmanr(y,p).statistic),
                         r2=float(r2_score(y,p)), mae=float(mean_absolute_error(y,p))))
    return pd.DataFrame(rows)

def binary(frame):
    if frame.label.nunique() < 2:
        return {'auroc_contrast': None, 'ap_contrast': None, 'status': 'one_class'}
    r, c, y = frame.probability_random_stratified, frame.probability_complete_cc, frame.label
    return {'auroc_contrast': float(roc_auc_score(y,r)-roc_auc_score(y,c)),
            'ap_contrast': float(average_precision_score(y,r)-average_precision_score(y,c)), 'status':'ok'}

def main():
    protocol = OUT / 'PROTOCOL.json'
    plan = json.loads(protocol.read_text())
    assert plan['status'] == 'RETROSPECTIVE_SENSITIVITY_PLAN_BEFORE_NEW_COMPUTATIONS'
    paths = [protocol, M/'evaluation_design_predictions.csv', M/'evaluation_design_metrics.csv',
             H/'walsh_split_predictions.csv', H/'moller_pairwise_genome_similarity.csv']
    d = pd.read_csv(paths[1], dtype={'cc':str,'host_id':str})
    assert not d.duplicated(['design','host_id','phage']).any()
    assert len(d) == 217*8*3
    assert d.groupby(['host_id','phage']).observed_od600.nunique().eq(1).all()
    recomputed = quantitative(d).set_index(['design','phage']).sort_index()
    recorded = pd.read_csv(paths[2]).set_index(['design','phage']).sort_index()
    assert np.allclose(recomputed[['spearman','r2','mae']],recorded[['spearman','r2','mae']],atol=1e-10)
    rows=[]
    for cc in sorted(d.cc.unique()):
        scores=quantitative(d[d.cc != cc]).groupby('design')[['spearman','r2','mae']].mean()
        contrast=scores.loc['random_kfold']-scores.loc['complete_cc_holdout']
        rows.append({'omitted_cc':cc, **{k:float(v) for k,v in contrast.items()}})
    influence=pd.DataFrame(rows)
    influence.to_csv(OUT/'moller_scoring_omission.csv', index=False)
    d['absolute_error']=(d.observed_od600-d.predicted_od600).abs()
    per_cc=d.groupby(['design','cc','phage']).absolute_error.mean().reset_index(name='mae')
    per_cc.to_csv(OUT/'per_lineage_phage_mae.csv', index=False)
    balanced=per_cc.groupby('design').mae.mean()
    cc_mae=per_cc.groupby(['cc','design']).mae.mean().unstack()
    hosts=d[['host_id','cc']].drop_duplicates()
    counts=hosts.cc.value_counts().sort_index()
    counts.rename_axis('cc').reset_index(name='n_hosts').to_csv(OUT/'lineage_counts.csv',index=False)
    walsh=pd.read_csv(paths[3],dtype={'cc':str})
    w=pd.DataFrame([{'omitted_cc':cc, **binary(walsh[walsh.cc != cc])} for cc in sorted(walsh.cc.unique())])
    w.to_csv(OUT/'walsh_scoring_omission.csv',index=False)
    pairs=pd.read_csv(paths[4],dtype={'reported_cc_a':str,'reported_cc_b':str})
    g=[]
    for cc in sorted(counts.index):
        b=pairs[(pairs.reported_cc_a != cc)&(pairs.reported_cc_b != cc)]
        labels=b.same_reported_cc.astype(int)
        s=b.sourmash_jaccard_k31_scaled1000
        g.append({'omitted_cc':cc,'same_cc_auc':float(roc_auc_score(labels,s)),
                  'median_similarity_contrast':float(s[labels==1].median()-s[labels==0].median())})
    genome=pd.DataFrame(g)
    genome.to_csv(OUT/'genome_lineage_omission.csv',index=False)
    result={
        'status':'COMPLETE', 'interpretation':'Retrospective fixed-prediction sensitivity, not refitted validation or new confidence intervals.',
        'input_text_lf_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        'headline_metrics_reproduced':True,
        'moller_omission_contrast_ranges':{k:[float(influence[k].min()),float(influence[k].max())] for k in ['spearman','r2','mae']},
        'moller_equal_cc_mae':balanced.to_dict(),
        'moller_cc_random_lower_mae_count':int((cc_mae.random_kfold<cc_mae.complete_cc_holdout).sum()),
        'moller_n_cc':len(counts), 'moller_largest_cc_n':int(counts.max()),
        'moller_inverse_cc_concentration':float(1/np.sum((counts/counts.sum())**2)),
        'walsh_omission_contrast_ranges':{k:[float(w[k].min()),float(w[k].max())] for k in ['auroc_contrast','ap_contrast']},
        'walsh_valid_omissions':int(w.status.eq('ok').sum()),
        'genome_omission_auc_range':[float(genome.same_cc_auc.min()),float(genome.same_cc_auc.max())]
    }
    (OUT/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
