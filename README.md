# Lineage-Aware Evaluation of Genome-Based Phage Susceptibility Prediction

This repository contains the scientific artifacts for a computational analysis of whether genome-based models can generalize *Staphylococcus aureus* bacteriophage-susceptibility predictions to previously unseen clonal lineages.

## Scientific result

The analysis shows that random isolate-level validation can substantially overstate generalization when related bacterial genomes occur in both training and test partitions. Using a fixed phenotype-blind 80% protein-pangenome representation and ridge pipeline, random cross-validation obtains macro Spearman 0.457 and $R^2=0.132$, while complete clonal-complex holdout obtains 0.190 and -0.237.

The repository also includes a phenotype-blind genomic-neighborhood audit, a separate 45-host assay-cohort split comparison, and a one-time outcome-blind evaluation of a frozen p0017 receptor rule on 30 recovered hosts. The separate assay cohort is not established to be genomically independent: cross-cohort similarity is high for several hosts (see below).

## Contents

- `data/`: compact frozen protocols, result summaries, prediction tables, and bootstrap outputs.
- `scripts/verify_results.py`: consistency checks for the headline results.
- `scripts/reproduce_ridge.py`: refits all original ridge predictions from the sparse protein-family matrix, checks stored predictions, reports training-mean baselines, and optionally runs all 20 predeclared random seeds.
- `scripts/run_lineage_sensitivity.py`: fixed-prediction lineage omission and equal-lineage error checks.
- `scripts/matched_fold_size_sensitivity.py`: random partitions with exactly the same test-fold sizes as grouped evaluation, across 20 predeclared seeds.
- `results/`: numerical reproduction and partition-sensitivity results, including every seed.
- `.gitignore`: explicit protection against accidentally committing raw reads, assemblies, caches, or credentials.

## Scope

The raw sequencing reads, assemblies, downloaded software caches, and temporary outputs are intentionally excluded. The data in this repository are compact scientific result artifacts derived from public bacterial datasets. No patient data are included.

## Reproduce headline checks

Use Python 3.11 with the recorded numerical environment:

```bash
python -m pip install -r requirements.txt
python scripts/verify_results.py
python scripts/verify_integrity.py
python scripts/reproduce_ridge.py --repeat-seeds
python scripts/run_lineage_sensitivity.py
python scripts/matched_fold_size_sensitivity.py
```

The script checks the reported random and lineage-held-out metrics, lineage effect size, genome-neighborhood AUC, and frozen recovery-validation metrics.

The refit script independently reconstructs 5,208 out-of-fold predictions (217 hosts x eight phages x three designs). Maximum discrepancy in the recorded environment is 2.3e-16. It starts from a 217 x 9,373 sparse presence/absence matrix, not from sequencing reads. Protein-family construction was global and phenotype-blind, so this is a transductive family dictionary, not a fully inductive pangenome workflow. Training-fold preprocessing consists of prevalence filtering to [0.02, 0.98] and `StandardScaler(with_mean=False)`; ridge uses alpha=10 and LSQR with an intercept. No duplicate-column collapse is applied by this analysis.

## Robustness and limitations

All additional sensitivity protocols explicitly disclose that headline outcomes were already known. They do not constitute new confirmatory validation. The original primary split is retained; no new seed is selected to replace it.

- Every single-CC scoring omission preserves the primary contrast. Equal-CC MAE is 0.157 for random versus 0.192 for grouped evaluation, favoring random in 13/15 CCs.
- Twenty additional balanced random partitions yield Spearman 0.428-0.500 and R2 0.073-0.185.
- Twenty random partitions matching grouped test-fold sizes (88, 50, 27, 26, 26) yield Spearman 0.430-0.507 and R2 0.072-0.227. Size matching does not control all differences in training composition.
- These ranges are descriptive, not confidence intervals. Existing paired CC-bootstrap intervals condition on fixed predictions and do not include model-refitting uncertainty. Walsh has only six CCs.
- Pair-bootstrap genome-similarity intervals in the original result JSON do not account adequately for shared-host dependence. Use the point estimates and explicitly labeled lineage-omission checks, not those intervals as independent-sample evidence.
- The 30-host receptor-rule evaluation is an internal recovery cohort with targeted-feature QC, not external/unseen-lineage validation. It failed its frozen success gate.

`data/ROBUSTNESS_AUDIT_PROTOCOL.json` is explicitly an abridged retrospective summary, not the original frozen bytes referred to by the hash in `ROBUSTNESS_AUDIT_RESULT.json`. The original scientific results are unchanged.

The new lineage-sensitivity manifest hashes text after canonical CRLF-to-LF normalization (`input_text_lf_sha256`) so the checks work across operating systems. Historical hashes retain their original conventions.

## Exact second-cohort reproduction and correction attempt

`python scripts/reproduce_walsh.py` refits the historical second-cohort predictions from the included feature and observation tables, checking host identities, labels, grouped inner selection, and outer CC separation. Both designs reproduce to numerical precision. **Only four features actually enter this model:** genome length, GC fraction, contig count, and N50. The other 142 of 146 candidate columns are entirely missing and are dropped by the original imputer. Historical metadata reporting a feature count of 146 counts candidate columns, not effective predictors. Therefore this is an assembly-summary comparison, not a receptor/defense-feature replication. `results/walsh_reproduction/RESULT.json` lists missing columns, effective fold dimensions and selection details; the original results remain unchanged.

`python scripts/run_lineage_remedy.py` reproduces the bounded retrospective correction experiment; `python scripts/summarize_lineage_remedy.py` verifies all 17,360 predictions and 80 per-phage summaries. It compares original ridge, equal-lineage-weighted ridge, within-lineage-centered ridge, isolate-mean and lineage-mean baselines in both original grouped fivefold and leave-one-CC-out designs. Under grouped fivefold, equal-CC MAE is respectively 0.191823, 0.187890, 0.190360, 0.172308 and 0.169318. The attempted within-lineage correction does not outperform the mean baselines. All models, protocols, predictions and negative results are retained in `results/lineage_remedy/`; no alpha or feature search is performed. See `data/LINEAGE_REMEDY_PROTOCOL.json` for the pre-computation plan and numerical amendment.

## Fixed genome-support diagnostic and cross-cohort provenance

`python scripts/test_genome_support.py` checks self-exclusion and train-only threshold behavior. `python scripts/genome_support_diagnostic.py` applies one fixed outcome-free support gate to the existing predictions: accept a query when its nearest training genome has similarity at least the fifth percentile of training leave-self-out nearest-neighbor similarities. The gate accepts 204/217 random-split hosts (94.0%) but 0/217 complete-CC-held-out hosts. Zero coverage is failure to provide predictions for that deployment shift, not evidence of accurate unseen-lineage prediction. No alternative threshold is selected. Protocol, all gate decisions, per-lineage coverage and summaries are retained.

`scripts/cross_cohort_genomes.py --assembly-root PATH` requires the original assembly layout and manifests (not included), plus sourmash 4.9.4 and Biopython. It recomputes k=31, scaled=1000 sketches for all 262 genomes. All 9,765 cross-cohort similarities and 262 assembly accessions/checksums are in `results/cross_cohort_genomes/`. Ten fixed within-Moller validation pairs reproduce exactly. Thirteen of 45 Walsh hosts have a nearest Moller Jaccard >=0.95; four reach >=0.99 (maximum 0.996524). These fixed descriptive thresholds flag close genomic relatives, not proven duplicate isolates. No hosts were excluded and no prediction was refit based on these findings. Separate accessions and assays must not be described as proof of independent genomes.

## Source attribution

Moller et al. (2021), *Genes Influencing Phage Host Range in Staphylococcus aureus on a Species-Wide Scale*, mSphere: https://doi.org/10.1128/mSphere.01263-20.

Walsh et al. (2023), *The host phylogeny determines viral infectivity and replication across Staphylococcus host species*, PLOS Pathogens: https://doi.org/10.1371/journal.ppat.1011433. Source data and code: https://doi.org/10.6084/m9.figshare.21642209.v1.

Derived tables retain source biological identifiers. Upstream sources retain their original terms. No claim is made that these assays or genomes were newly collected for this analysis.

## Interpretation

These analyses are not a clinical phage-selection system and must not be used to make treatment decisions. The project studies validation design and biological generalization in structured genomic data.
