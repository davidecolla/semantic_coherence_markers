from config import configuration
from tqdm import tqdm
import os
import re

from experiments.experiment1_gpt2 import build_leave_one_out_dataset_speeches_trump
from utils.ngrams import compute_perplexity, train_ngrams
from utils import utils

# ----------------------------------------- #
#           Experiment 1 - Ngrams
# ----------------------------------------- #

def run_experiment(order):
    # Jurafsky P.40
    utils.print_title(str(order) + "-Grams speeches Intra-Trump", )

    build_leave_one_out_dataset_speeches_trump()

    vocabulary_fp = '../resources/data/experiment1/input/all_transcripts.txt'
    build_vocabulary_file(vocabulary_fp)

    base_path = configuration.experiment1_train_data_base_path
    subject_interview_perplexity = {}

    # rally or interview
    for training_category in tqdm(os.listdir(base_path), desc="Testing category"):

        # e.g. /{PRJ_HOME}/resources/data/input/experiment1/datasets/rally/
        training_category_folder = base_path + training_category + "/"
        subject_interview_perplexity[training_category] = {}

        category_training_set = ""

        # /{PRJ_HOME}/resources/data/input/experiment1/datasets/rally/1/
        for leave_out_speech in os.listdir(training_category_folder):
            leave_out_speech_folder = training_category_folder + leave_out_speech + "/"
            train_set_file_path = leave_out_speech_folder + "/train.txt"
            test_set_file_path = leave_out_speech_folder + "/test.txt"

            # Read training file content
            with open(train_set_file_path, 'r') as f:
                text = f.read()

            # # ------------------------- Train N-grams ------------------------- #
            model = train_ngrams(text, order, vocabulary_fp)
            # # ----------------------------------------------------------------- #

            # # ------------------------------------------------------------ #
            # # ------------------------- Test Set ------------------------- #
            # # ------------------------------------------------------------ #
            # Computing PPL on the leave out file
            with open(test_set_file_path, 'r') as f:
                file_content = f.read()
            file_content = re.sub(r"\[.*?]", "", file_content)

            # Store test set text
            category_training_set += file_content + "\n"

            # Compute test set (leave out speech) perplexity
            avg_ppl, dev_std = compute_perplexity(file_content, order, model)

            if training_category not in subject_interview_perplexity[training_category]:
                subject_interview_perplexity[training_category][training_category] = {}
            subject_interview_perplexity[training_category][training_category][leave_out_speech] = {"perplexity": avg_ppl, "dev.std": dev_std}
        # End training intra-category files


        # # ---------------------------------------------------------------------------------- #
        # # ------------------------- Run test on the other category ------------------------- #
        # # ---------------------------------------------------------------------------------- #

        # Retrain ngrams
        model = train_ngrams(category_training_set, order, vocabulary_fp)

        # interview
        for test_category in os.listdir(base_path):
            if test_category != training_category:
                test_category_folder = base_path + test_category + "/"
                subject_interview_perplexity[training_category][test_category] = {}

                # /{PRJ_HOME}/resources/data/input/experiment1/datasets/interview/1/
                for leave_out_speech in os.listdir(test_category_folder):
                    leave_out_speech_folder = test_category_folder + leave_out_speech + "/"
                    test_set_file_path = leave_out_speech_folder + "/test.txt"

                    with open(test_set_file_path, 'r') as f:
                        file_content = f.read()
                    file_content = re.sub(r"\[.*?]", "", file_content)


                    # Compute test set (leave out speech) perplexity
                    avg_ppl, dev_std = compute_perplexity(file_content, order, model)

                    if test_category not in subject_interview_perplexity[training_category]:
                        subject_interview_perplexity[training_category][test_category] = {}
                    subject_interview_perplexity[training_category][test_category][leave_out_speech] = {
                                                                                    "perplexity": avg_ppl,
                                                                                             "dev.std": dev_std}
        # # ---------------------------------------------------------------------------------- #
    # End category

    # Writing results
    out_file = configuration.experiment1_output_base_path + str(order) + "-grams.csv"
    print("Writing results to: " + out_file)
    write_csv(subject_interview_perplexity, out_file)

def write_csv(subject_interview_perplexity, out_file):
    csv = "Subject,Fine-tuning,Speech Category,Transcript ID,AVG Perplexity,Dev. STD\n"

    for category_train in subject_interview_perplexity:
        for category_test in subject_interview_perplexity[category_train]:
                for speech_id in subject_interview_perplexity[category_train][category_test]:
                    csv += "Trump," + category_train + "," + category_test + "," + speech_id + "," + \
                           str(subject_interview_perplexity[category_train][category_test][speech_id]["perplexity"]) + \
                           "," + \
                           str(subject_interview_perplexity[category_train][category_test][speech_id]["dev.std"])
                    csv += "\n"


    with open(out_file, 'w') as f:
        f.write(csv)

def build_vocabulary_file(vocabulary_fp):
    all_text = ""
    for category in os.listdir(configuration.experiment1_data_base_path):
        if not category.startswith(".") and category != "all_transcripts.txt":
            category_folder = configuration.experiment1_data_base_path + category + "/"

            for category_transcript in os.listdir(category_folder):
                category_file = category_folder + category_transcript
                file_content = open(category_file, 'r').read()
                all_text += file_content + "\n"

    open(vocabulary_fp, 'w').write(all_text)


if __name__ == '__main__':
    run_experiment(2)
