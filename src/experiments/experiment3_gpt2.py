from config import configuration
from tqdm import tqdm
import os
from os import path
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from gpt2 import GPT2, gpt2_train
from gpt2.GPT2 import GPT2
from utils.utils import print_title, compute_text_avg_perplexity_window
import json


# ----------------------------------------- #
# Experiment 3 Trial 1 - PPL difference - GPT2
# ----------------------------------------- #
# For each subject s we compute the difference between two ppl scores:
# - ppl_{own} = ppl score provided in leave-one-subject out setting for subject in the same class of s
# - ppl_{other} = ppl score provided by the model trained on the 'other' class

def run_experiment(training_epochs):

    print_title("Experiment 3 - GPT-2 fine-tuning epochs: " + str(training_epochs))

    # --- Build paths for cookie test ---
    # "../resources/data/experiment3/input/pitt/"
    data_base_path = configuration.experiment3_data_base_path
    models_base_path = configuration.experiment3_gpt2_models_base_path

    build_all_data()

    # ------------------------------------------------------- #
    #          1. Train Single LM on ALL Control group        #
    # ------------------------------------------------------- #
    if training_epochs == 0:
        print("Skipping training of models: training epochs are set to 0.")
        all_control_model_dir = "gpt2"
    else:
        all_control_train_file_path = data_base_path + "ALL/control.txt"
        all_control_model_dir = models_base_path + "ALL/control_" + configuration.experiment3_gpt2_model_name + "_" + str(training_epochs) + "EP/"
        if path.exists(all_control_model_dir) and len(os.listdir(all_control_model_dir)) > 0:
            print("Skip training for ALL dementia")
            print("Folder " + all_control_model_dir + " already exists and is not empty.")
        else:
            print("Start training for ALL control group.")
            gpt2 = GPT2()
            gpt2.train_and_save(all_control_model_dir,
                                all_control_train_file_path,
                                configuration.experiment3_gpt2_model_name,
                                str(configuration.experiment3_gpt2_batch_size),
                                str(training_epochs))
            print("Training has been completed!")

    # ------------------------------------------------------- #
    #          2. Train Single LM on ALL Dementia group       #
    # ------------------------------------------------------- #
    if training_epochs == 0:
        print("Skipping training of models: training epochs are set to 0.")
        all_dementia_model_dir = "gpt2"
    else:
        all_dementia_train_file_path = data_base_path + "ALL/dementia.txt"
        all_dementia_model_dir = models_base_path + "ALL/dementia_" + configuration.experiment3_gpt2_model_name + "_" + str(training_epochs) + "EP/"
        if path.exists(all_dementia_model_dir) and len(os.listdir(all_dementia_model_dir)) > 0:
            print("Skip training for ALL dementia")
            print("Folder " + all_dementia_model_dir + " already exists and is not empty.")
        else:
            print("Start training for ALL dementia group.")
            gpt2 = GPT2()
            gpt2.train_and_save(all_dementia_model_dir,
                                all_dementia_train_file_path,
                                configuration.experiment3_gpt2_model_name,
                                str(configuration.experiment3_gpt2_batch_size),
                                str(training_epochs))
            print("Training has been completed!")


    # ------------------------------------------------------- #
    #  3. Train LMs on control group in leave_one_out_setting #
    # ------------------------------------------------------- #
    build_leave_one_out("control")
    control_subjects = get_partition_subjects_dictionary("control")
    control_data_dir = data_base_path + "control/"

    if training_epochs == 0:
        print("Skipping training of models: training epochs are set to 0.")
    else:
        loo_control_model_dir = models_base_path + "control/"

        for leave_out_subject_id in tqdm(control_subjects, desc="Training control"):
            train_set_file_path = control_data_dir + "datasets/" + leave_out_subject_id + "/train.txt"
            train_model_output_dir = loo_control_model_dir + leave_out_subject_id + "_" + configuration.experiment3_gpt2_model_name + "_" + str(training_epochs) + "EP/"
            os.makedirs(train_model_output_dir, exist_ok=True)

            # --- Train neural model for psycho_test_name ---
            if path.exists(train_model_output_dir) and len(os.listdir(train_model_output_dir)) > 0:
                print("Skip training for " + leave_out_subject_id)
                print("Folder " + train_model_output_dir + " already exists and is not empty.")
            else:
                print("Start training for " + leave_out_subject_id + "...")

                # Build training arguments dictionary
                args = {"output_dir": train_model_output_dir,
                        "model_name_or_path": configuration.experiment3_gpt2_model_name,
                        "do_train": "y",
                        "train_file": train_set_file_path,
                        "per_device_train_batch_size": configuration.experiment3_gpt2_batch_size,
                        "num_train_epochs": training_epochs}

                # Write training arguments file
                with open(configuration.experiment3_gpt2_arguments_file, 'w') as f:
                    json.dump(args, f)
                print("Training arguments:\n\n" + str(args))

                # Run training with training file parameter
                gpt2_train.train_model(["", configuration.experiment3_gpt2_arguments_file])

                print("Training has been completed!")


    # ------------------------------------------------------- #
    # 4. Train LMs on dementia group in leave_one_out_setting #
    # ------------------------------------------------------- #
    build_leave_one_out("dementia")
    dementia_subjects = get_partition_subjects_dictionary("dementia")
    dementia_data_dir = data_base_path + "dementia/"

    if training_epochs == 0:
        print("Skipping training of models: training epochs are set to 0.")
    else:
        loo_dementia_model_dir = models_base_path + "dementia/"

        for leave_out_subject_id in tqdm(dementia_subjects, desc="Training dementia"):
            train_set_file_path = dementia_data_dir + "datasets/" + leave_out_subject_id + "/train.txt"
            train_model_output_dir = loo_dementia_model_dir + leave_out_subject_id + "_" + configuration.experiment3_gpt2_model_name + "_" + str(training_epochs) + "EP/"
            os.makedirs(train_model_output_dir, exist_ok=True)

            # --- Train neural model for psycho_test_name ---
            if path.exists(train_model_output_dir) and len(os.listdir(train_model_output_dir)) > 0:
                print("Skip training for " + leave_out_subject_id)
                print("Folder " + train_model_output_dir + " already exists and is not empty.")
            else:
                print("Start training for " + leave_out_subject_id + "...")

                # Build training arguments dictionary
                args = {"output_dir": train_model_output_dir,
                        "model_name_or_path": configuration.experiment3_gpt2_model_name,
                        "do_train": "y",
                        "train_file": train_set_file_path,
                        "per_device_train_batch_size": configuration.experiment3_gpt2_batch_size,
                        "num_train_epochs": training_epochs}

                # Write training arguments file
                with open(configuration.experiment3_gpt2_arguments_file, 'w') as f:
                    json.dump(args, f)
                print("Training arguments:\n\n" + str(args))

                # Run training with training file parameter
                gpt2_train.train_model(["", configuration.experiment3_gpt2_arguments_file])

                print("Training has been completed!")

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                      Models training have been completed.                                        #
    # ---------------------------------------------------------------------------------------------------------------- #
    # In the following we will exploit the above trained models to compute the perplexity scores for all subjects in
    # the control/dementia group.

    # Final dictionary. Will be in the following form:
    # {
    #   "control": {
    #       "1": {
    #           "0": {
    #               "p_control": 12345,
    #               "dev.std_control": 6789,
    #               "p_dementia": 54321,
    #               "dev.std_dementia": 9876,
    #           },
    #           "1": {
    #               "p_control": 12345,
    #               "dev.std_control": 6789,
    #               "p_dementia": 54321,
    #               "dev.std_dementia": 9876,
    #           },
    #           ...
    #       },
    #       ...
    #   },
    #   "dementia": {
    #       ...
    #   }
    # }
    group_patient_ppl_dictionary = {"control": {}, "dementia": {}}
    # Load ALL control model
    model_all_control = GPT2LMHeadModel.from_pretrained(all_control_model_dir)
    tokenizer_all_control = GPT2TokenizerFast.from_pretrained(all_control_model_dir)
    # Load ALL dementia model
    model_all_dementia = GPT2LMHeadModel.from_pretrained(all_dementia_model_dir)
    tokenizer_all_dementia = GPT2TokenizerFast.from_pretrained(all_dementia_model_dir)

    # ------------------------------- #
    #  5. Deal with control subjects  #
    # ------------------------------- #
    for leave_out_subject_id in tqdm(control_subjects, desc="Perplexity control"):
        # Load above trained leave-out model
        if training_epochs == 0:
            train_model_output_dir = "gpt2"
        else:
            train_model_output_dir = loo_control_model_dir + leave_out_subject_id + "_" + configuration.experiment3_gpt2_model_name + "_" + str(training_epochs) + "EP/"
        model = GPT2LMHeadModel.from_pretrained(train_model_output_dir)
        tokenizer = GPT2TokenizerFast.from_pretrained(train_model_output_dir)

        # Compute the perplexity for the leave-out subject
        for interview_id in control_subjects[leave_out_subject_id]:

            # Load interview text
            interview_file_path = control_data_dir + leave_out_subject_id + "-" + interview_id + ".txt"
            with open(interview_file_path, 'r') as f:
                file_content = f.read()

            # Get perplexity for the current interview from the leave_out model
            avg_ppl = compute_text_avg_perplexity_window(file_content, tokenizer, model, window_size=20)

            # Get perplexity for the current interview from the ALL dementia model
            avg_ppl_dementia = compute_text_avg_perplexity_window(file_content, tokenizer_all_dementia,
                                                                  model_all_dementia, window_size=20)

            # Add subject's scores
            if leave_out_subject_id not in group_patient_ppl_dictionary["control"]:
                group_patient_ppl_dictionary["control"][leave_out_subject_id] = {}
            if interview_id not in group_patient_ppl_dictionary["control"][leave_out_subject_id]:
                group_patient_ppl_dictionary["control"][leave_out_subject_id][interview_id] = {}
            group_patient_ppl_dictionary["control"][leave_out_subject_id][interview_id]["p_control"] = avg_ppl
            group_patient_ppl_dictionary["control"][leave_out_subject_id][interview_id]["p_dementia"] = avg_ppl_dementia

    # ------------------------------- #
    #  5. Deal with dementia subjects #
    # ------------------------------- #
    for leave_out_subject_id in tqdm(dementia_subjects, desc="Perplexity dementia"):
        # Load above trained leave-out model
        if training_epochs == 0:
            train_model_output_dir = "gpt2"
        else:
            train_model_output_dir = loo_dementia_model_dir + leave_out_subject_id + "_" + configuration.experiment3_gpt2_model_name + "_" + str(training_epochs) + "EP/"
        model = GPT2LMHeadModel.from_pretrained(train_model_output_dir)
        tokenizer = GPT2TokenizerFast.from_pretrained(train_model_output_dir)

        # Compute the perplexity for the leave-out subject
        for interview_id in dementia_subjects[leave_out_subject_id]:

            # Load interview text
            interview_file_path = dementia_data_dir + leave_out_subject_id + "-" + interview_id + ".txt"
            with open(interview_file_path, 'r') as f:
                file_content = f.read()

            # Get perplexity for the current interview
            avg_ppl = compute_text_avg_perplexity_window(file_content, tokenizer, model, window_size=20)

            # Get perplexity for the current interview from the ALL control model
            avg_ppl_control = compute_text_avg_perplexity_window(file_content, tokenizer_all_control,
                                                                 model_all_control, window_size=20)

            # Add subject's scores
            if leave_out_subject_id not in group_patient_ppl_dictionary["dementia"]:
                group_patient_ppl_dictionary["dementia"][leave_out_subject_id] = {}
            if interview_id not in group_patient_ppl_dictionary["dementia"][leave_out_subject_id]:
                group_patient_ppl_dictionary["dementia"][leave_out_subject_id][interview_id] = {}
            group_patient_ppl_dictionary["dementia"][leave_out_subject_id][interview_id]["p_dementia"] = avg_ppl
            group_patient_ppl_dictionary["dementia"][leave_out_subject_id][interview_id]["p_control"] = avg_ppl_control


    out_file = configuration.experiment3_output_base_path + "GPT-2_EP=" + str(training_epochs) + ".csv"
    print("Writing results to: " + out_file)
    write_csv_pitt(group_patient_ppl_dictionary, out_file)


def build_all_data():
    data_base_path = configuration.experiment3_data_base_path
    controls = get_partition_subjects_dictionary("control")
    ad = get_partition_subjects_dictionary("dementia")

    control_texts = ""
    ad_texts = ""

    for file in os.listdir(data_base_path + "control/"):
        if not file.startswith(".") and file != "datasets":
            subject = file.split("-")[0]
            file_content = open(data_base_path + "control/" + file, 'r').read().replace("\n\n", "\n")
            if subject in controls:
                control_texts += file_content + "\n"

    for file in os.listdir(data_base_path + "dementia/"):
        if not file.startswith(".") and file != "datasets":
            subject = file.split("-")[0]
            file_content = open(data_base_path + "dementia/" + file, 'r').read().replace("\n\n", "\n")
            if subject in ad:
                ad_texts += file_content + "\n"

    os.makedirs(data_base_path + "ALL/", exist_ok=True)
    print(data_base_path + "ALL/")
    open(data_base_path + "ALL/control.txt", "w").write(control_texts)
    open(data_base_path + "ALL/dementia.txt", "w").write(ad_texts)


def get_partition_subjects_dictionary(partition):
    base_path = configuration.experiment3_data_base_path + partition + "/"
    # --- Build subjects dictionary ---
    subject_id_interview_id = {}
    for file_name in os.listdir(base_path):
        if file_name != "datasets" and file_name != "." and file_name != "..":
            subject_id = file_name.replace(".txt", "").split("-")[0]
            interview_id = file_name.replace(".txt", "").split("-")[1]

            if subject_id not in subject_id_interview_id:
                subject_id_interview_id[subject_id] = []
            subject_id_interview_id[subject_id].append(interview_id)

    # Remove patients with a single interview
    for k, v in list(subject_id_interview_id.items()):
        if len(v) <= 1:
            del subject_id_interview_id[k]

    return subject_id_interview_id


def build_leave_one_out(partition):
    base_path = configuration.experiment3_data_base_path + partition + "/"
    # --- Build subjects dictionary ---
    subject_id_interview_id = {}
    for file_name in os.listdir(base_path):
        if file_name != "datasets" and file_name != "." and file_name != "..":
            subject_id = file_name.replace(".txt", "").split("-")[0]
            interview_id = file_name.replace(".txt", "").split("-")[1]

            if subject_id not in subject_id_interview_id:
                subject_id_interview_id[subject_id] = []
            subject_id_interview_id[subject_id].append(interview_id)

    # Remove patients with a single interview
    for k, v in list(subject_id_interview_id.items()):
        if len(v) <= 1:
            del subject_id_interview_id[k]

    print("Dementia subjects# " + str(len(subject_id_interview_id)))
    # ---

    # Leave-One-Out: leave one subject out
    for leave_out_subject_id in subject_id_interview_id:
        train_set_base_dir = base_path + "datasets/" + leave_out_subject_id + "/"
        train_set_file_path = train_set_base_dir + "train.txt"
        os.makedirs(train_set_base_dir, exist_ok=True)

        # --- Build train set for psycho_test_name and leave_out_subject_id ---
        # Building train set for leave_out_subject_id means concatenate texts of
        # all interviews except for leave_out_subject_id
        # print("Building train set for leave_out: " + leave_out_subject_id)
        build_train_set_for_subject_leave_one_out(leave_out_subject_id, subject_id_interview_id, train_set_file_path,
                                                  base_path)


def build_train_set_for_subject_leave_one_out(leave_out_subject_id, subject_id_interview_id, train_set_file_path, base_path):
    # --- Build train set for current psycho_test by joining whole group files ---
    train_set_text = ""

    for leave_in_subject_id in subject_id_interview_id:

        if leave_out_subject_id != leave_in_subject_id:

            for interview_id in subject_id_interview_id[leave_in_subject_id]:
                interview_file_path = base_path + leave_in_subject_id + "-" + interview_id + ".txt"

                with open(interview_file_path, 'r') as f:
                    # Read file content and get rid of trash
                    file_content = f.read()

                train_set_text += file_content + "\n"

            # End loop on interviews
    # End loop on subjects
    # Here train_set_text contains the train_set text. We are now able to write text on file.
    with open(train_set_file_path, 'w') as f:
        f.write(train_set_text)


def write_csv_pitt(group_patient_ppl_dictionary, out_file):
    csv = "Group Id, Subject Id, Interview ID,P_Control, Dev.STD_Control, P_Dementia, Dev.STD_Dementia\n"

    # Save control group
    for partition in group_patient_ppl_dictionary:
        for patient in group_patient_ppl_dictionary[partition]:
            for interview in group_patient_ppl_dictionary[partition][patient]:
                csv += partition + "," + patient + "," + interview + "," + \
                       str(group_patient_ppl_dictionary[partition][patient][interview]["p_control"]) + "," + \
                       str(group_patient_ppl_dictionary[partition][patient][interview]["dev.std_control"]) + "," + \
                       str(group_patient_ppl_dictionary[partition][patient][interview]["p_dementia"]) + "," + \
                       str(group_patient_ppl_dictionary[partition][patient][interview]["dev.std_dementia"]) + "\n"
    csv += "\n"
    with open(out_file, 'w') as f:
        f.write(csv)


if __name__ == '__main__':
    run_experiment(5)
