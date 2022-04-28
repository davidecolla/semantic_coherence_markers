import statistics
from experiments.experiment2_gpt2 import build_speech_dataset
from config import configuration
from tqdm import tqdm
import os
import re
from utils.ngrams import train_ngrams, compute_perplexity
from utils.utils import print_title


# -------------------------------------------- #
# Experiment 2.1 - Ngrams - speaker vs speaker
# -------------------------------------------- #

def run_experiment(order):

    print_title("Experiment 2 - " + str(order) + "-grams")

    speech_perplexity = {}

    # {PRJ_HOME}/resources/data/experiment2/input/obama/
    for speaker in os.listdir(configuration.experiment2_speeches_base_path):
        if speaker != "." and speaker != ".." and speaker != "datasets" and not speaker.startswith("."):

            print("Generating Dataset...")
            build_speech_dataset(speaker)

            print("Start training...")

            train_data_directory = configuration.experiment2_train_data_base_path
            base_path = train_data_directory + speaker + "/"
            train_set_file_path = base_path + "train.txt"

            text = open(train_set_file_path, 'r').read()
            model = train_ngrams(text, order)


            # ------------------------------------------------------------------------------- #
            #                   Compute perplexity across different speaker                   #
            # ------------------------------------------------------------------------------- #
            # Compute the perplexity of the model trained above for interviews of the subject leve_out_subject_id.


            for other_speaker in tqdm(os.listdir(train_data_directory), desc=("Computing ppl with model of " + speaker)):
                if other_speaker != "." and other_speaker != ".." and other_speaker != speaker and not other_speaker.startswith("."):

                    other_speaker_base_path = configuration.experiment2_speeches_base_path + other_speaker + "/"
                    other_speaker_ppls = []

                    for speaker_speech_file_name in os.listdir(other_speaker_base_path):
                        other_speaker_test_file_path = other_speaker_base_path + speaker_speech_file_name

                        subject_interview = open(other_speaker_test_file_path, 'r').read()
                        subject_interview = re.sub(r"\[.*?]", "", subject_interview)

                        avg_ppl, dev_std = compute_perplexity(subject_interview, order, model)
                        other_speaker_ppls.append(avg_ppl)


                    # ---------------------------------------------------------------------------------------- #
                    #
                    # "biden":{
                    #   "obama": {"perplexity": avg_ppl, "dev.std": dev_std},
                    #   "trump": {"perplexity": avg_ppl, "dev.std": dev_std},
                    #   ...
                    #  },
                    #  ...
                    # },
                    avg_ppl = statistics.mean(other_speaker_ppls)
                    dev_std = statistics.stdev(other_speaker_ppls)
                    if speaker not in speech_perplexity:
                        speech_perplexity[speaker] = {}
                    if other_speaker not in speech_perplexity[speaker]:
                        speech_perplexity[speaker][other_speaker] = {}
                    speech_perplexity[speaker][other_speaker] = {"perplexity": avg_ppl, "dev.std": dev_std}

    print("=" * 50)
    print("Training has been completed!")
    out_file = configuration.experiment2_output_base_path + str(order) + "-grams.csv"
    print("Writing results to " + out_file)
    write_csv(speech_perplexity, out_file)

def write_csv(speeches_perplexity, out_file):

    csv = ""
    for speaker in speeches_perplexity:
        csv += "," + speaker + ","
    csv += "\n"

    for speaker1 in speeches_perplexity:
        csv += speaker1 + ","
        for speaker2 in speeches_perplexity:
            if speaker2 == speaker1:
                csv += ",,"
            else:
                ppl = speeches_perplexity[speaker1][speaker2]["perplexity"]
                devstd = speeches_perplexity[speaker1][speaker2]["dev.std"]
                csv += str(ppl) + "," + str(devstd) + ","
        csv += "\n"

    with open(out_file, 'w') as f:
        f.write(csv)


if __name__ == '__main__':
    run_experiment(2)