"""Descriptive cross-cohort genomic provenance audit; never reads phenotype values."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from importlib.metadata import version
import pandas as pd
from Bio import SeqIO
import sourmash

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assembly-root', type=Path, required=True)
    args = parser.parse_args()
    protocol = ROOT/'data/CROSS_COHORT_GENOME_PROTOCOL.json'
    assert json.loads(protocol.read_text())['status'].endswith('BEFORE_CROSS_COHORT_COMPUTATION')
    cohorts = {
        'moller': (ROOT/'data/pangenome_hosts.csv', args.assembly_root/'data/hostbio_phase1/moller/annotation_manifest.json'),
        'walsh': (ROOT/'data/walsh_split_predictions.csv', args.assembly_root/'data/walsh_2023_isp/benchmark/assembly_provenance.json')}
    sketches = {}; inputs = []; ids = {}
    for cohort, (host_file, manifest_file) in cohorts.items():
        ids[cohort] = pd.read_csv(host_file, usecols=['biosample']).biosample.tolist()
        manifest = {a['biosample']: a for a in json.loads(manifest_file.read_text())['assemblies']}
        for b in ids[cohort]:
            entry = manifest[b]
            relative = entry['files']['fasta']['path'].replace('\\', '/')
            fasta = args.assembly_root/relative
            mh = sourmash.MinHash(n=0, ksize=31, scaled=1000)
            opener = gzip.open if fasta.suffix == '.gz' else open
            length = 0; contigs = 0
            with opener(fasta, 'rt') as handle:
                for record in SeqIO.parse(handle, 'fasta'):
                    mh.add_sequence(str(record.seq), force=True)
                    length += len(record.seq); contigs += 1
            assert len(mh) > 0
            sketches[b] = mh
            inputs.append({'cohort':cohort, 'biosample':b, 'assembly_accession':entry['assembly_accession'],
                           'relative_path':relative, 'sha256':hashlib.sha256(fasta.read_bytes()).hexdigest(),
                           'bases':length, 'contigs':contigs, 'sketch_size':len(mh)})
        print(f'{cohort}: {len(ids[cohort])} genomes sketched', flush=True)
    assert len(ids['moller']) == 217 and len(ids['walsh']) == 45
    old = pd.read_csv(ROOT/'data/moller_pairwise_genome_similarity.csv').head(10)
    errors = [abs(sketches[r.biosample_a].jaccard(sketches[r.biosample_b])-r.sourmash_jaccard_k31_scaled1000) for r in old.itertuples()]
    assert max(errors) < 1e-12, errors
    pairs = pd.DataFrame([{'walsh_biosample':w, 'moller_biosample':m, 'jaccard':sketches[w].jaccard(sketches[m])}
                          for w in ids['walsh'] for m in ids['moller']])
    nearest = pairs.loc[pairs.groupby('walsh_biosample').jaccard.idxmax()].sort_values('jaccard', ascending=False)
    out = ROOT/'results/cross_cohort_genomes'; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inputs).to_csv(out/'assembly_inputs.csv', index=False)
    pairs.to_csv(out/'all_pairs.csv', index=False)
    nearest.to_csv(out/'nearest_cross_cohort.csv', index=False)
    result = {'status':'COMPLETE', 'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
              'sourmash_version':version('sourmash'), 'n_pairs':len(pairs),
              'validation_max_absolute_error':max(errors), 'maximum_jaccard':float(pairs.jaccard.max()),
              'walsh_hosts_with_nearest_at_least_095':int((nearest.jaccard >= .95).sum()),
              'walsh_hosts_with_nearest_at_least_099':int((nearest.jaccard >= .99).sum()),
              'interpretation':'High similarity flags require provenance investigation; no phenotype-based exclusions made.'}
    (out/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
