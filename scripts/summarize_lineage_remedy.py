"""Summarize all prespecified per-phage outcomes without refitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error

OUT=Path(__file__).resolve().parents[1]/'results/lineage_remedy'
d=pd.read_csv(OUT/'predictions.csv',dtype={'cc':str})
assert not d.duplicated(['design','model','biosample','phage']).any()
assert len(d)==2*5*217*8
assert d.groupby(['biosample','phage'])['observed'].nunique().eq(1).all()
rows=[]
for (design,model,phage),block in d.groupby(['design','model','phage']):
    error=(block.observed-block.predicted).abs()
    rows.append({'design':design,'model':model,'phage':phage,
                 'spearman':float(spearmanr(block.observed,block.predicted).statistic),
                 'r2':float(r2_score(block.observed,block.predicted)),
                 'mae':float(mean_absolute_error(block.observed,block.predicted)),
                 'equal_cc_mae':float(error.groupby(block.cc).mean().mean())})
metrics=pd.DataFrame(rows)
metrics.to_csv(OUT/'per_phage_metrics.csv',index=False)
summary=json.loads((OUT/'RESULT.json').read_text())
for (design,model),block in metrics.groupby(['design','model']):
    for name in ['spearman','r2','mae','equal_cc_mae']:
        assert np.isclose(block[name].mean(),summary['designs'][design]['models'][model][name],atol=1e-12)
print('All 17360 predictions unique; observed outcomes consistent; all 80 per-phage rows reconcile with macro results.')
