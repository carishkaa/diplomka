import argparse
from typing import TypedDict
import warnings
import numpy as np
from statsmodels.stats.correlation_tools import cov_nearest
from scipy.stats import multivariate_normal, norm
import bindata as bnd
import pandas as pd

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)

# https://research.wu.ac.at/ws/portalfiles/portal/18952613/document.pdf
def generate_multivariate_binary(num_vars, num_samples, marginal_probs, corr_matrix, threshold_drugs_per_sample: tuple=None):
    if len(marginal_probs) != num_vars:
        raise ValueError("Number of variables does not match the length of the means vector.")

    if corr_matrix.shape != (num_vars, num_vars):
        raise ValueError("Correlation matrix dimensions do not match the number of variables.")

    commonprob = bnd.bincorr2commonprob(margprob=marginal_probs, bincorr=corr_matrix)
    sigma = bnd.commonprob2sigma(commonprob)
    if not is_pos_def(sigma):
        sigma = cov_nearest(sigma)

    µ, Σ = norm.ppf(marginal_probs), sigma
    mvn = multivariate_normal(µ, Σ, allow_singular=True)
    sample_norm = mvn.rvs(size=num_samples)

    sample_metadata: Metadata = { 'mu': µ, 'sigma': Σ, 'mvn': mvn, 'sample_norm': sample_norm }
    if threshold_drugs_per_sample is None:
        # From the paper: convert to binary by thresholding the normal samples at 0.
        return sample_norm > 0, sample_metadata

    # Convert to binary by choosing top k drugs per sample.
    min_drugs_per_sample, max_drugs_per_sample = threshold_drugs_per_sample
    drugs_per_sample = np.random.choice(range(min_drugs_per_sample, max_drugs_per_sample + 1), size=num_samples) # TODO maybe not uniformly?
    sample_binary = np.zeros_like(sample_norm)
    for i in range(num_samples):
        indices = np.argpartition(sample_norm[i], -drugs_per_sample[i])[-drugs_per_sample[i]:]
        sample_binary[i][indices] = 1

    return sample_binary, sample_metadata

class Metadata(TypedDict):
    mu: np.ndarray
    sigma: np.ndarray
    mvn: any
    sample_norm: np.ndarray

def filter_rows(sample, mu, sigma, target_numvars):
    """ Filter rows with number of ones in the range [3, 7] and add new rows if necessary. """
    sum_ones_per_row = np.sum(sample, axis=1)
    sample = sample[(sum_ones_per_row >= 3) & (sum_ones_per_row <= 7)]

    while True:
        batch_size = target_numvars - sample.shape[0]
        if batch_size == 0:
            break

        new_sample = multivariate_normal(mu, sigma, allow_singular=True).rvs(size=batch_size) > 0

        sum_ones_per_row = np.sum(new_sample, axis=1) if new_sample.ndim == 2 else np.sum(new_sample)
        new_sample = new_sample[(sum_ones_per_row >= 3) & (sum_ones_per_row <= 7)]
        sample = np.concatenate((sample, new_sample), axis=0)
    
    return sample

def binary2capsules(sample, drug_names):
    capsules = pd.DataFrame(sample, columns=drug_names)
    capsules['drug_names'] = capsules.apply(lambda x: [drug_names[i] for i in range(len(drug_names)) if x.iloc[i] == 1], axis=1)
    capsules = capsules.drop(columns=drug_names)
    return capsules

def generate_dosages(capsule, dosages):
    generated_dosages = []
    for drug in capsule:
        min_dosage = dosages[dosages['Drug Names'] == drug]['MIN units [pcs]'].values[0]
        max_dosage = dosages[dosages['Drug Names'] == drug]['MAX units [pcs]'].values[0]
        generated_dosages.append(np.random.randint(min_dosage, max_dosage))
    return generated_dosages

def load_dosages():
    dosages = pd.read_csv("drugs_dosing.csv", sep=';')
    dosages = dosages[['Drug Names', 'MIN units [pcs]', 'MAX units [pcs]']]
    dosages['Average units [pcs]'] = (dosages['MIN units [pcs]'] + dosages['MAX units [pcs]']) / 2
    return dosages

def generator(num_vars, num_samples, marginals, corr_matrix, cnames, dosages, threshold_drugs_per_sample):
    sample, _ = generate_multivariate_binary(num_vars, num_samples, marginals, corr_matrix, threshold_drugs_per_sample)
    capsules = binary2capsules(sample, cnames)
    capsules['SEQN'] = capsules.index
    capsules['dosages'] = capsules.apply(lambda x: generate_dosages(x['drug_names'], dosages), axis=1)

    res_capsules = capsules[['SEQN', 'drug_names', 'dosages']]
    return res_capsules

def main(num_vars, num_samples, marginals_filename, corr_matrix_filename, sorted_names_filename, output_filename, threshold_drugs_per_sample):
    means = np.loadtxt(marginals_filename, delimiter=",")
    corr_matrix = np.loadtxt(corr_matrix_filename, delimiter=",")
    dosages = load_dosages()
    drug_names = pd.read_csv(sorted_names_filename, header=None)[0].to_list()

    df = generator(num_vars, num_samples, means, corr_matrix, drug_names, dosages, threshold_drugs_per_sample)
    df.to_csv(output_filename, sep=';', index=False)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patients generator", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--num-vars", type=int, help="Number of variables (drugs)", default=40)
    parser.add_argument("-n", "--num-samples", type=int, help="Number of samples (patients) to generate", default=1000)
    parser.add_argument("-p", "--marginal-probas", type=argparse.FileType('r'), help="CSV file containing marginal probabilities of each drug", metavar=('MARGINAL_PROBAS_FILE'), required=True)
    parser.add_argument("-c", "--corr-matrix", type=argparse.FileType('r'), help="CSV file containing the correlation matrix of the drugs", metavar=('CORR_MATRIX_FILE'), required=True)
    parser.add_argument("-s", "--sorted-names", type=argparse.FileType('r'), help="CSV file containing the drug names in the same order as marginals and correlation matrix", metavar=('SORTED_NAMES_FILE'))
    parser.add_argument("-d", "--drugs-per-sample", type=int, nargs=2, metavar=('MIN', 'MAX'), help="Minimum and maximum number of drugs per sample. If not provided, it will not be limited and vary naturally.")
    parser.add_argument("-o", "--output", type=str, help="Output file name", default='generated_capsules_with_dosages.csv', metavar=('OUTPUT_FILE'))

    args = parser.parse_args()

    with warnings.catch_warnings(record=True) as recorded_warnings:
        main(args.num_vars, args.num_samples, args.marginal_probas, args.corr_matrix, args.sorted_names, args.output, args.drugs_per_sample)

    print("Generated successfully and saved to \"" + args.output + "\".")
