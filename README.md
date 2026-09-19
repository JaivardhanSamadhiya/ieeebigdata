# Lineage-Aware Evaluation of Genome-Based Phage Susceptibility Prediction

This repository contains the scientific artifacts for a computational analysis of whether genome-based models can generalize *Staphylococcus aureus* bacteriophage-susceptibility predictions to previously unseen clonal lineages.

## Scientific result

The analysis shows that random isolate-level validation can substantially overstate generalization when related bacterial genomes occur in both training and test partitions. Using a fixed phenotype-blind 80% protein-pangenome representation and ridge pipeline, random cross-validation obtains macro Spearman 0.457 and $R^2=0.132$, while complete clonal-complex holdout obtains 0.190 and -0.237.

The repository also includes a phenotype-blind genomic-neighborhood audit, an independent 45-host cohort split comparison, and a one-time outcome-blind evaluation of a frozen p0017 receptor rule on 30 recovered hosts.

## Contents

- `data/`: compact frozen protocols, result summaries, prediction tables, and bootstrap outputs.
- `scripts/verify_results.py`: consistency checks for the headline results.
- `.gitignore`: explicit protection against accidentally committing raw reads, assemblies, caches, or credentials.

## Scope

The raw sequencing reads, assemblies, downloaded software caches, and temporary outputs are intentionally excluded. The data in this repository are compact scientific result artifacts derived from public bacterial datasets. No patient data are included.

## Reproduce headline checks

Use Python with `pandas` installed:

```bash
python scripts/verify_results.py
```

The script checks the reported random and lineage-held-out metrics, lineage effect size, genome-neighborhood AUC, and frozen recovery-validation metrics.

## Interpretation

These analyses are not a clinical phage-selection system and must not be used to make treatment decisions. The project studies validation design and biological generalization in structured genomic data.
