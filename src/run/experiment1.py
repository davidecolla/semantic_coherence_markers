import utils.utils
from experiments import experiment1_gpt2, experiment1_ngrams


def main():
    utils.utils.print_title("Experiment 1")

    # Run experiment 1 for GPT2 models
    for epochs in[0,5,10,20,30]:
        experiment1_gpt2.run_experiment(epochs)

    # Run experiment 1 for N-gram models
    for order in [2,3,4,5]:
        experiment1_ngrams.run_experiment(order)

if __name__ == '__main__':
    main()