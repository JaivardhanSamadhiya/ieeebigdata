"""Verify recorded scientific inputs and protocol hashes without fitting."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    result=json.loads((ROOT/'data/lineage_sensitivity/RESULT.json').read_text())
    for name,expected in result['input_text_lf_sha256'].items():
        content=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(content).hexdigest()==expected, f'Input hash mismatch: {name}'
    match=json.loads((ROOT/'results/matched_fold_size_sensitivity.json').read_text())
    assert sha(ROOT/'data/MATCHED_FOLD_SIZE_PROTOCOL.json')==match['protocol_sha256']
    assert len(match['all_seeds'])==20
    assert [r['seed'] for r in match['all_seeds']]==list(range(2026092000,2026092020))
    assert match['test_sizes']==[88,50,27,26,26]
    refit=json.loads((ROOT/'results/ridge_reproduction.json').read_text())
    assert sha(ROOT/'data/pangenome_presence_absence.npz')==refit['matrix_sha256']
    assert sha(ROOT/'data/REFIT_SENSITIVITY_PROTOCOL.json')==refit['protocol_sha256']
    assert len(refit['retrospective_random_seed_sensitivity'])==20
    assert all(d['max_prediction_discrepancy']<1e-6 for d in refit['designs'].values())
    for directory,protocol in [('lineage_remedy','LINEAGE_REMEDY_PROTOCOL.json'),
                               ('genome_support','GENOME_SUPPORT_PROTOCOL.json'),
                               ('cross_cohort_genomes','CROSS_COHORT_GENOME_PROTOCOL.json')]:
        recorded=json.loads((ROOT/'results'/directory/'RESULT.json').read_text())
        assert sha(ROOT/'data'/protocol)==recorded['protocol_sha256'],directory
    gate=json.loads((ROOT/'results/genome_support/RESULT.json').read_text())
    assert sha(ROOT/'results/genome_support/gates_before_error_scoring.csv')==gate['gate_before_error_scoring_sha256']
    assert gate['designs']['complete_cc_holdout']['n_accepted']==0
    assert gate['designs']['complete_cc_holdout']['accepted_mae'] is None
    walsh=json.loads((ROOT/'results/walsh_reproduction/RESULT.json').read_text())
    for name,expected in walsh['input_text_lf_sha256'].items():
        assert hashlib.sha256((ROOT/'data'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected,name
    assert len(walsh['available_columns'])==4 and len(walsh['entirely_missing_columns'])==142
    assert all(r['max_prediction_error']<1e-10 for r in walsh['designs'].values())
    print('Scientific input hashes, all extension protocols, complete seed sets, and recorded prediction checks pass.')

if __name__=='__main__': main()
