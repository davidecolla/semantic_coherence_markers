import utils.utils
from config import configuration
from experiments import experiment2_gpt2, experiment2_ngrams
import os
import numpy as np
from tabulate import tabulate
from rpy2.robjects import DataFrame, FloatVector, IntVector
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

def main():
    utils.utils.print_title("Experiment 2")

    # Run experiment 2 for GPT2 models
    for epochs in[0,5,20,20,30]:
        experiment2_gpt2.run_experiment(epochs)

    # Run experiment 2 for N-gram models
    for order in [2,3,4,5]:
        experiment2_ngrams.run_experiment(order)

    # ICC
    compute_icc()

def compute_icc():

    # Install IRR R package
    utils = importr('utils')
    utils.chooseCRANmirror(ind=1)  # select the first mirror in the list
    utils.install_packages('irr')

    model_subject_scores = {}

    for file_name in os.listdir(configuration.experiment2_output_base_path):
        file_path = configuration.experiment2_output_base_path + file_name
        model_name = file_name.replace(".csv", "")

        lines = open(file_path, 'r').read().split("\n")

        subjects = lines[0].split(",")

        for line in lines[1:]:
            split = line.split(",")
            for i,score in enumerate(split):
                if i < len(subjects) and subjects[i] != "" and score != "":

                    if model_name not in model_subject_scores:
                        model_subject_scores[model_name] = {}
                    if subjects[i] not in model_subject_scores[model_name]:
                        model_subject_scores[model_name][subjects[i]] = []
                    model_subject_scores[model_name][subjects[i]].append(float(score))

    # Compute ICC by leveraging R irr package.
    # Thanks to the python package rpy2 and the conda R management we are able to run R commands
    # through python.
    results = []
    for model in model_subject_scores:
        values = []
        for subject in model_subject_scores[model]:
            scores = model_subject_scores[model][subject]
            values.extend([float(x) for x in scores])

        r_irr = importr("irr")
        m = robjects.r.matrix(FloatVector(values), nrow=8, ncol=7, byrow=True)

        result = r_irr.icc(m, model="twoway", type="consistency", unit="single")

        icc_val = np.asarray(result[6])[0]  # icc_val now holds the icc value
        results.append([model, "{:.2f}".format(icc_val)])

    print(tabulate(results, headers=['Model', 'ICC score'], tablefmt='orgtbl'))


if __name__ == '__main__':
    main()