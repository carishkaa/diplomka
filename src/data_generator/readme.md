# Data Generator

## Run
```bash
python3 main.py -v 40 -n 1000 -p feat_probs.csv -c correlation_matrix.csv --sorted-names drug_names.csv -d 3 7
```

```
  -h, --help            show this help message and exit
  -v NUM_VARS, --num-vars NUM_VARS
                        Number of variables (drugs) (default: 40)
  -n NUM_SAMPLES, --num-samples NUM_SAMPLES
                        Number of samples (patients) to generate (default: 1000)
  -p MARGINAL_PROBAS_FILE, --marginal-probas MARGINAL_PROBAS_FILE
                        CSV file containing marginal probabilities of each drug (default: None)
  -c CORR_MATRIX_FILE, --corr-matrix CORR_MATRIX_FILE
                        CSV file containing the correlation matrix of the drugs (default: None)
  -s SORTED_NAMES_FILE, --sorted-names SORTED_NAMES_FILE
                        CSV file containing the drug names in the same order as marginals and correlation matrix (default: None)
  -d MIN MAX, --drugs-per-sample MIN MAX
                        Minimum and maximum number of drugs per sample. If not provided, it will not be limited and vary
                        naturally. (default: None)
  -o OUTPUT_FILE, --output OUTPUT_FILE
                        Output file name (default: generated_capsules_with_dosages.csv)
```

Based on https://research.wu.ac.at/ws/portalfiles/portal/18952613/document.pdf, but if min and max number of drugs per sample provided, we just pick top `k = [min, max]` elements from normal sample during converting to binary. 

todo description, library requirements