import argparse
import warnings
import numpy as np
from statsmodels.stats.correlation_tools import cov_nearest
from scipy.stats import multivariate_normal, norm
import bindata as bnd
import pandas as pd

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)

# https://research.wu.ac.at/ws/portalfiles/portal/18952613/document.pdf
def generate_multivariate_binary(num_vars, num_samples, marginal_probs, corr_matrix):
    if len(marginal_probs) != num_vars:
        raise ValueError("Number of variables does not match the length of the means vector.")
    
    if corr_matrix.shape != (num_vars, num_vars):
        raise ValueError("Correlation matrix dimensions do not match the number of variables.")
    
    commonprob = bnd.bincorr2commonprob(margprob=marginal_probs, 
                                            bincorr=corr_matrix)
    sigma = bnd.commonprob2sigma(commonprob)
    if not is_pos_def(sigma):
        not_pos_def_sigma = sigma
        sigma = cov_nearest(sigma)

    µ, Σ = norm.ppf(marginal_probs), sigma
    mvn = multivariate_normal(µ, Σ, allow_singular=True)
    sample = mvn.rvs(size=num_samples)
    # sample_binary = sample > 0
    # uniformly from 3 to 7 

    # get 3 maximal elements of sample
    sample_binary = np.zeros_like(sample)
    for i in range(sample.shape[0]):
        num_of_drugs = np.random.choice([3, 4, 5, 6, 7, 8], p=[0.92, 0.07, 0.0045, 0.0039, 0.0015, 0.0001])
        indices = np.argpartition(sample[i], -num_of_drugs)[-num_of_drugs:]
        sample_binary[i][indices] = 1

    return sample_binary, µ, Σ, mvn.rvs(size=num_samples)

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

def load_means(file_path):
    return np.loadtxt(file_path, delimiter=",")

def load_corr_matrix(file_path):
    return np.loadtxt(file_path, delimiter=",")

def load_dosages(file_path):
    dosages = pd.read_csv(file_path, sep=';')
    dosages = dosages[['Drug Names', 'MIN units [pcs]', 'MAX units [pcs]']]
    dosages['Average units [pcs]'] = (dosages['MIN units [pcs]'] + dosages['MAX units [pcs]']) / 2
    return dosages

def generator(num_vars, num_samples, marginals, corr_matrix, cnames, dosages):
    sample, mu, sigma, _ = generate_multivariate_binary(num_vars, num_samples, marginals, corr_matrix)
    sample = filter_rows(sample, mu, sigma, num_samples) # number of drugs [3, 7]
    capsules = binary2capsules(sample, cnames)
    capsules['SEQN'] = capsules.index
    capsules['dosages'] = capsules.apply(lambda x: generate_dosages(x['drug_names'], dosages), axis=1)

    res_capsules = capsules[['SEQN', 'drug_names', 'dosages']]
    return res_capsules

def main(num_vars, num_samples, marginals_filename, corr_matrix_filename, cnames_filename, dosages_filename, output_filename):
    means = load_means(marginals_filename)
    corr_matrix = load_corr_matrix(corr_matrix_filename)
    dosages = load_dosages(dosages_filename)
    drug_names = pd.read_csv(cnames_filename, header=None)[0].to_list()

    df = generator(num_vars, num_samples, means, corr_matrix, drug_names, dosages)
    df.to_csv(output_filename, sep=';', index=False)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input arguments.")

    parser.add_argument("-v", "--num_vars", type=int, help="Number of variables", default=40)
    parser.add_argument("-n", "--num_samples", type=int, help="Number of samples to generate", default=1000)
    parser.add_argument("-m", "--marginals", type=argparse.FileType('r'), help="CSV file containing marginal probabilities")
    parser.add_argument("-c", "--corr_matrix", type=argparse.FileType('r'), help="CSV file containing the correlation matrix")
    parser.add_argument("--names_file", type=argparse.FileType('r'), help="CSV file containing the drug names in the same order as marginals and correlation matrix")
    parser.add_argument("-d", "--dosages", type=argparse.FileType('r'), help="Dosages")
    default_output_file = 'generated_capsules_with_dosages.csv'

    args = parser.parse_args()

    with warnings.catch_warnings(record=True) as recorded_warnings:
        main(args.num_vars, args.num_samples, args.marginals_file, args.corr_matrix_file, args.cnames_file, args.dosages_file, default_output_file)

    print("Generated successfully and saved to \"" + default_output_file + "\".")