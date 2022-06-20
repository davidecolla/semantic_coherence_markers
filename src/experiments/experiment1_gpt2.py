from config import configuration
from gpt2 import gpt2_train
from utils import utils
from tqdm import tqdm
import os
import json
import re
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ----------------------------------------- #
#             Experiment 1 - GPT2
# ----------------------------------------- #
from utils.utils import compute_text_avg_perplexity_window


def run_experiment(training_epochs):
    utils.print_title("Train GPT Model - Leave One Out Speeches Trump Epochs:" + str(training_epochs))
    print("Training epochs:", training_epochs)
    print("Generating Dataset...")
    build_leave_one_out_dataset_speeches_trump()

    print("Start training...")
    leave_one_out_base_path = configuration.experiment1_train_data_base_path
    leave_one_out_model_base_folder = configuration.experiment1_gpt2_models_base_path

    subject_interview_perplexity = {}


    # rally speeches or interviews
    for category in tqdm(os.listdir(leave_one_out_base_path)):
        # e.g. /{PRJ_HOME}/resources/data/input/experiment1/datasets/rally/
        category_folder = leave_one_out_base_path + category + "/"
        subject_interview_perplexity[category] = {}

        # /{PRJ_HOME}/resources/data/input/experiment1/datasets/rally/1/
        for leave_out_speech_file_name in tqdm(os.listdir(category_folder)):
            leave_out_speech_id = leave_out_speech_file_name.replace(".txt", "")
            train_set_file_path = category_folder + leave_out_speech_file_name + "/train.txt"
            test_set_file_path = category_folder + leave_out_speech_file_name + "/test.txt"


            model_out_folder = leave_one_out_model_base_folder + category + "_" + leave_out_speech_id + "_epochs=" + str(training_epochs) + "/"

            if training_epochs == 0:
                model_out_folder = configuration.experiment1_gpt2_model_name
            else:
                os.makedirs(model_out_folder, exist_ok=True)
                if os.path.exists(model_out_folder) and len(os.listdir(model_out_folder)) > 0:
                    print("Skip training for " + category + " - " + leave_out_speech_id)
                    print("Folder " + model_out_folder + " already exists and is not empty.")
                else:
                    print("Loading train file from:" + train_set_file_path)
                    print("Output dir:" + model_out_folder)
                    print("Start training for " + category + " - " + leave_out_speech_id)
                    # ------------------------------------------------------------------------------- #
                    #                            Train the GPT2 Model                                 #
                    # ------------------------------------------------------------------------------- #
                    # Train the GPT2 model on texts obtained by concatenating all transcriptions for all subjects except
                    # leve_out_subject_id.

                    # Build training arguments dictionary
                    args = {"output_dir": model_out_folder,
                            "model_name_or_path": configuration.experiment1_gpt2_model_name,
                            "do_train": "y",
                            "train_file": train_set_file_path,
                            "per_device_train_batch_size": configuration.experiment1_gpt2_batch_size,
                            "num_train_epochs": training_epochs}

                    # Write training arguments file
                    with open(configuration.experiment1_gpt2_arguments_file, 'w') as f:
                        json.dump(args, f)
                    print("Training arguments:\n\n" + str(args))

                    # Run training with training file parameter
                    gpt2_train.train_model(["", configuration.experiment1_gpt2_arguments_file])

                    print("Training has been completed!")

            # ------------------------------------------------------------------------------- #
            #                            Compute perplexity                                   #
            # ------------------------------------------------------------------------------- #
            # Compute the perplexity of the model trained above for interviews of the subject leve_out_subject_id.
            # Load GPT2 Model

            # Test on category's leave-out-speech
            model = GPT2LMHeadModel.from_pretrained(model_out_folder)
            tokenizer = GPT2TokenizerFast.from_pretrained(model_out_folder)

            print("Computing perplexity for subject: " + category + " - " + leave_out_speech_id)

            with open(test_set_file_path, 'r') as f:
                subject_interview = f.read()

            avg_ppl = compute_text_avg_perplexity_window(subject_interview, tokenizer, model)

            if category not in subject_interview_perplexity[category]:
                subject_interview_perplexity[category][category] = {}
            subject_interview_perplexity[category][category][leave_out_speech_id] = {"perplexity": avg_ppl}


            # --- Test on other categories ---
            for other_category in os.listdir(leave_one_out_base_path):

                if other_category != category and other_category != "datasets" and not other_category.startswith("."):
                    other_category_folder = leave_one_out_base_path + other_category + "/"
                    subject_interview_perplexity[category][other_category] = {}

                    for leave_in_speech_file_name in tqdm(os.listdir(other_category_folder), desc=other_category):
                        leave_in_speech_file_path = other_category_folder + leave_in_speech_file_name
                        leave_in_speech_id = leave_in_speech_file_name.replace(".txt", "")


                        print("Computing perplexity with model: " + category + " - " + leave_out_speech_id + " on transcript " + other_category + " - " + leave_in_speech_id)

                        with open(leave_in_speech_file_path, 'r') as f:
                            subject_interview = f.read()
                        subject_interview = re.sub(r"\[.*?]", "", subject_interview)

                        avg_ppl = compute_text_avg_perplexity_window(subject_interview, tokenizer, model)

                        if leave_in_speech_id not in subject_interview_perplexity[category][other_category]:
                            subject_interview_perplexity[category][other_category][leave_in_speech_id] = {}

                        subject_interview_perplexity[category][other_category][leave_in_speech_id][leave_out_speech_id] = {"perplexity": avg_ppl}

    print("Training has been completed!")
    print("="*50)

    out_file = configuration.experiment1_output_base_path + "GPT-2_EP=" + str(training_epochs) + ".csv"
    print("Writing results to " + out_file)
    write_csv(subject_interview_perplexity, out_file)


def write_csv(subject_interview_perplexity, eval_file):

    csv = "Subject,Fine-tuning,Speech Category,Transcript ID,AVG Perplexity\n"

    for category_train in subject_interview_perplexity:
        for category_test in subject_interview_perplexity[category_train]:
            if category_test == category_train:
                for speech_id in subject_interview_perplexity[category_train][category_test]:
                    csv += "Trump," + category_train + "," + category_test + "," + speech_id + "," + \
                           str(subject_interview_perplexity[category_train][category_test][speech_id]["perplexity"]) + "\n"
            else:

                for speech_id in subject_interview_perplexity[category_train][category_test]:
                    ppls = []
                    for predictor_speech_id in subject_interview_perplexity[category_train][category_test][speech_id]:
                        ppls.append(subject_interview_perplexity[category_train][category_test][speech_id][predictor_speech_id]["perplexity"])

                    csv += "Trump," + category_train + "," + category_test + "," + speech_id + "," + \
                           str(np.mean(ppls)) + "\n"
    with open(eval_file, 'w') as f:
        f.write(csv)


def build_leave_one_out_dataset_speeches_trump():
    base_path = configuration.experiment1_data_base_path
    out_base_path = configuration.experiment1_train_data_base_path

    for category in os.listdir(base_path):
        if category != "datasets" and category != "all_transcripts.txt" and not category.startswith("."):
            category_folder = base_path + category + "/"

            for leave_out_speech_file_name in os.listdir(category_folder):
                leave_out_speech_id = leave_out_speech_file_name.replace(".txt", "")
                out_folder = out_base_path + category + "/" + leave_out_speech_id + "/"
                os.makedirs(out_folder, exist_ok=True)

                # ----------------------------------------
                # Build test set for leave_out_subject_id
                with open(category_folder + leave_out_speech_file_name, 'r') as f:
                    file_content = f.read()
                with open(out_folder + "test.txt", 'w') as f:
                    f.write(file_content)
                # ----------------------------------------
                # Build train set for leave_out_subject_id
                leave_out_train = ""
                for leave_in_speech_file_name in os.listdir(category_folder):
                    if leave_in_speech_file_name != leave_out_speech_file_name:
                        with open(category_folder + leave_in_speech_file_name, 'r') as f:
                            file_content = f.read()
                        leave_out_train += file_content + "\n"

                with open(out_folder + "train.txt", 'w') as f:
                    f.write(leave_out_train)


if __name__ == '__main__':
    run_experiment(5)
