import utils.utils
from experiments import experiment2_gpt2, experiment2_ngrams


def main():
    utils.utils.print_title("Experiment 2")

    # Run experiment 2 for GPT2 models
    for epochs in[0,5,20,20,30]:
        experiment2_gpt2.run_experiment(epochs)

    # Run experiment 2 for N-gram models
    for order in [2,3,4,5]:
        experiment2_ngrams.run_experiment(order)

if __name__ == '__main__':
    main()