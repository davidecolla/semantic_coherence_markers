import statistics
from gpt2 import gpt2_train
from config import configuration
from tqdm import tqdm
import os
from os import path
import json
import re

from utils.utils import print_title, compute_text_avg_perplexity_window
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# ----------------------------------------- #
#             Experiment 2 - GPT2
# ----------------------------------------- #

def run_experiment(training_epochs):

    print_title("Experiment 2 - GPT-2 fine-tuning epochs: " + str(training_epochs))

    speech_perplexity = {}

    # {PRJ_HOME}/resources/data/experiment2/input/obama/
    for speaker in os.listdir(configuration.experiment2_speeches_base_path):
        if speaker != "." and speaker != ".." and speaker != "datasets" and not speaker.startswith("."):

            print("Generating Dataset...")
            build_speech_dataset(speaker)

            print("Start training...")
            leave_one_out_model_base_folder = configuration.experiment2_gpt2_models_base_path

            train_data_directory = configuration.experiment2_train_data_base_path
            base_path = train_data_directory + speaker + "/"

            train_set_file_path = base_path + "train.txt"
            model_out_folder = leave_one_out_model_base_folder + speaker + "-epochs=" + \
                                   str(training_epochs) + "-model=" + configuration.experiment2_gpt2_model_name + "/"

            # Train GPT2 Model if needed
            if path.exists(model_out_folder) and len(os.listdir(model_out_folder)) > 0:
                print("Skip training for speaker " + speaker)
                print("Folder " + model_out_folder + " already exists and is not empty.")
            else:
                os.makedirs(model_out_folder, exist_ok=True)
                print("Loading train file from:" + train_set_file_path)
                print("Output dir:" + model_out_folder)
                print("Start training for speaker " + speaker)
                # ------------------------------------------------------------------------------- #
                #                            Train the GPT2 Model                                 #
                # ------------------------------------------------------------------------------- #
                # Train the GPT2 model on texts obtained by concatenating all transcriptions for all subjects except
                # leave_out_speech_id.

                # Build training arguments dictionary
                args = {"output_dir": model_out_folder,
                        "model_name_or_path": configuration.experiment2_gpt2_model_name,
                        "do_train": "y",
                        "train_file": train_set_file_path,
                        "per_device_train_batch_size": configuration.experiment2_gpt2_batch_size,
                        "num_train_epochs": training_epochs}

                # Write training arguments file
                with open(configuration.experiment2_gpt2_arguments_file, 'w') as f:
                    json.dump(args, f)
                print("Training arguments:\n\n" + str(args))

                # Run training with training file parameter
                gpt2_train.train_model(["", configuration.experiment2_gpt2_arguments_file])

                print("Training has been completed!")

            # ------------------------------------------------------------------------------- #
            #                   Compute perplexity across different speaker                   #
            # ------------------------------------------------------------------------------- #
            # Compute the perplexity of the model trained above on other speaker's speeches.
            # Load GPT2 Model
            model = GPT2LMHeadModel.from_pretrained(model_out_folder)
            tokenizer = GPT2TokenizerFast.from_pretrained(model_out_folder)


            for other_speaker in tqdm(os.listdir(train_data_directory), desc=("Computing ppl with model of " + speaker)):
                if other_speaker != "." and other_speaker != ".." and other_speaker != speaker and not other_speaker.startswith("."):

                    other_speaker_base_path = configuration.experiment2_speeches_base_path + other_speaker + "/"
                    other_speaker_ppls = []

                    for speaker_speech_file_name in os.listdir(other_speaker_base_path):
                        other_speaker_test_file_path = other_speaker_base_path + speaker_speech_file_name

                        subject_interview =  open(other_speaker_test_file_path, 'r').read()
                        subject_interview = re.sub(r"\[.*?]", "", subject_interview)

                        avg_ppl = compute_text_avg_perplexity_window(subject_interview, tokenizer, model)

                        other_speaker_ppls.append(float(avg_ppl))

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
    out_file = configuration.experiment2_output_base_path + "GPT-2_EP=" + str(training_epochs) + ".csv"
    print("Writing results to " + out_file)
    write_csv(speech_perplexity, out_file)


def write_csv(speeches_perplexity,eval_file):
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

    with open(eval_file, 'w') as f:
        f.write(csv)


def build_speech_dataset(speaker):

    base_path = configuration.experiment2_speeches_base_path + speaker + "/"
    out_base_path = configuration.experiment2_test_data_base_path + speaker + "/"
    train_data_directory = configuration.experiment2_train_data_base_path + speaker + "/"
    os.makedirs(train_data_directory, exist_ok=True)

    leave_one_out_datasets = {}
    speeches_text = ""
    for leave_out_speech_file_name in tqdm(os.listdir(base_path)):

        leave_out_speech_file_path = base_path + leave_out_speech_file_name
        leave_out_speech_id = leave_out_speech_file_name.replace(".txt", "")
        out_folder = out_base_path + leave_out_speech_id + "/"
        os.makedirs(out_folder, exist_ok=True)
        leave_one_out_datasets[leave_out_speech_file_name] = {}

        # ----------------------------------------
        # Build test set for leave_out_speech_file_name
        test_file_path = out_folder + "test.txt"
        with open(leave_out_speech_file_path, 'r') as f:
            file_content = f.read()
        with open(test_file_path, 'w') as f:
            f.write(file_content)
        speeches_text += file_content + "\n"
        # ----------------------------------------
        # Build train set for leave_out_speech_file_name
        leave_out_train = ""
        for leave_in_file_name in os.listdir(base_path):
            if leave_in_file_name != leave_out_speech_file_name:
                leave_in_file_path = base_path + leave_in_file_name

                with open(leave_in_file_path, 'r') as f:
                    file_content = f.read()
                leave_out_train += file_content + "\n"

        train_file_path = out_folder + "train.txt"
        with open(train_file_path, 'w') as f:
            f.write(leave_out_train)

        speeches_text = speeches_text.replace("\n\n", "\n")
        open(train_data_directory + "train.txt", "w").write(speeches_text)

if __name__ == '__main__':
    run_experiment(5)