from config import configuration
from tqdm import tqdm
import os

from experiments.experiment3_gpt2 import build_leave_one_out, get_partition_subjects_dictionary, build_all_data, \
    preprocess_pitt_corpus_data
from utils.ngrams import train_ngrams, compute_perplexity
from utils.utils import print_title


# ---------------------------------------------- #
# Experiment 3 Trial 1 - N-Grams PPL difference
# ---------------------------------------------- #
def run_experiment(order):
    print_title("Experiment 3 - " + str(order) + "-grams")

    # --- Build paths for cookie test ---
    # "../resources/data/experiment3/input/pitt/"
    data_base_path = configuration.experiment3_data_base_path

    # Preprocessing CHAT files
    preprocess_pitt_corpus_data()

    # Generate Concatenation of all files to train the "other" LM
    build_all_data()

    # Build a single vocabulary to train n-grams
    vocabulary_fp = '../resources/data/experiment3/input/all_transcripts.txt'
    build_vocabulary_file(vocabulary_fp)

    # ------------------------------------------------------- #
    #          1. Train Single LM on ALL Control group        #
    # ------------------------------------------------------- #
    all_control_train_file_path = data_base_path + "ALL/control.txt"

    print("Start training for ALL control group.")
    with open(all_control_train_file_path, 'r') as f:
        text = f.read()

    model_all_control = train_ngrams(text, order, vocabulary_fp)
    print("Training has been completed!")

    # ------------------------------------------------------- #
    #          2. Train Single LM on ALL AD group             #
    # ------------------------------------------------------- #
    all_dementia_train_file_path = data_base_path + "ALL/dementia.txt"

    print("Start training for ALL control group.")
    with open(all_dementia_train_file_path, 'r') as f:
        text = f.read()

    model_all_dementia = train_ngrams(text, order, vocabulary_fp)
    print("Training has been completed!")

    # ------------------------------------------------------- #
    #  3. Train LMs on control group in leave_one_out_setting #
    # ------------------------------------------------------- #
    build_leave_one_out("control")
    control_subjects = get_partition_subjects_dictionary("control")
    control_leave_out_models = {}
    control_data_dir = data_base_path + "control/"

    # For each subject s in the control group, train a LM on the concatenation of
    # all the transcripts from patients belonging to the control group except for s
    for leave_out_subject_id in tqdm(control_subjects, desc="Training control"):
        train_set_file_path = control_data_dir + "datasets/" + leave_out_subject_id + "/train.txt"

        # --- Train ngrams model for subject leave_out_subject_id ---
        with open(train_set_file_path, 'r') as f:
            text = f.read()
        leave_one_out_model = train_ngrams(text, order, vocabulary_fp)

        # Store the model in the dictionary
        control_leave_out_models[leave_out_subject_id] = leave_one_out_model

    # ------------------------------------------------------- #
    # 4. Train LMs on dementia group in leave_one_out_setting #
    # ------------------------------------------------------- #
    build_leave_one_out("dementia")
    dementia_subjects = get_partition_subjects_dictionary("dementia")
    dementia_leave_out_models = {}
    dementia_data_dir = data_base_path + "dementia/"

    # For each subject s in the AD group, train a LM on the concatenation of
    # all the transcripts from patients belonging to the AD group except for s
    for leave_out_subject_id in tqdm(dementia_subjects, desc="Training dementia"):
        train_set_file_path = dementia_data_dir + "datasets/" + leave_out_subject_id + "/train.txt"

        # --- Train ngrams model for subject leave_out_subject_id ---
        with open(train_set_file_path, 'r') as f:
            text = f.read()
        leave_one_out_model = train_ngrams(text, order, vocabulary_fp)

        # Store the model in the dictionary
        dementia_leave_out_models[leave_out_subject_id] = leave_one_out_model


    # ---------------------------------------------------------------------------------------------------------------- #
    #                                      Models training have been completed.                                        #
    # ---------------------------------------------------------------------------------------------------------------- #
    # In the following we will exploit the above trained models to compute the perplexity scores for all subjects in
    # the control/dementia group.

    # Teh final dictionary will be in the following form:
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

    # ------------------------------- #
    #  5. Deal with control subjects  #
    # ------------------------------- #
    #
    # For each subject s in the Control group, load the LMs acquired on ALL AD group and on the control group
    # by excluding s; compute the perplexity for s through such models; store the scores in our final dictionary.
    for leave_out_subject_id in tqdm(control_subjects, desc="Perplexity control"):
        # Load above trained leave-out model
        model = control_leave_out_models[leave_out_subject_id]

        # Compute the perplexity for the leave-out subject
        for interview_id in control_subjects[leave_out_subject_id]:

            # Load interview text, the test set
            interview_file_path = control_data_dir + leave_out_subject_id + "-" + interview_id + ".txt"
            with open(interview_file_path, 'r') as f:
                file_content = f.read()

            # Get perplexity for the current interview from the leave_out model
            avg_ppl, dev_std = compute_perplexity(file_content, order, model)

            # Get perplexity for the current interview from the ALL dementia model
            avg_ppl_dementia, dev_std_dementia = compute_perplexity(file_content, order, model_all_dementia)


            # Add subject's scores
            if leave_out_subject_id not in group_patient_ppl_dictionary["control"]:
                group_patient_ppl_dictionary["control"][leave_out_subject_id] = {}
            if interview_id not in group_patient_ppl_dictionary["control"][leave_out_subject_id]:
                group_patient_ppl_dictionary["control"][leave_out_subject_id][interview_id] = {}
            group_patient_ppl_dictionary["control"][leave_out_subject_id][interview_id]["p_control"] = avg_ppl
            group_patient_ppl_dictionary["control"][leave_out_subject_id][interview_id]["dev.std_control"] = dev_std
            group_patient_ppl_dictionary["control"][leave_out_subject_id][interview_id]["p_dementia"] = avg_ppl_dementia
            group_patient_ppl_dictionary["control"][leave_out_subject_id][interview_id]["dev.std_dementia"] = dev_std_dementia

    # ------------------------------- #
    #  5. Deal with dementia subjects #
    # ------------------------------- #
    #
    # For each subject s in the AD group, load the LMs acquired on ALL Control group and on the AD group
    # by excluding s; compute the perplexity for s through such models; store the scores in our final dictionary.
    for leave_out_subject_id in tqdm(dementia_subjects, desc="Perplexity dementia"):
        # Load above trained leave-out model
        model = dementia_leave_out_models[leave_out_subject_id]

        # Compute the perplexity for the leave-out subject
        for interview_id in dementia_subjects[leave_out_subject_id]:

            # Load interview text
            interview_file_path = dementia_data_dir + leave_out_subject_id + "-" + interview_id + ".txt"
            with open(interview_file_path, 'r') as f:
                file_content = f.read()

                # Get perplexity for the current interview from the leave_out model
                avg_ppl, dev_std = compute_perplexity(file_content, order, model)

                # Get perplexity for the current interview from the ALL control model
                avg_ppl_control, dev_std_control = compute_perplexity(file_content, order, model_all_control)


            # Add subject's scores
            if leave_out_subject_id not in group_patient_ppl_dictionary["dementia"]:
                group_patient_ppl_dictionary["dementia"][leave_out_subject_id] = {}
            if interview_id not in group_patient_ppl_dictionary["dementia"][leave_out_subject_id]:
                group_patient_ppl_dictionary["dementia"][leave_out_subject_id][interview_id] = {}
            group_patient_ppl_dictionary["dementia"][leave_out_subject_id][interview_id]["p_dementia"] = avg_ppl
            group_patient_ppl_dictionary["dementia"][leave_out_subject_id][interview_id][
                "dev.std_dementia"] = dev_std
            group_patient_ppl_dictionary["dementia"][leave_out_subject_id][interview_id][
                "p_control"] = avg_ppl_control
            group_patient_ppl_dictionary["dementia"][leave_out_subject_id][interview_id][
                "dev.std_control"] = dev_std_control

    out_file = configuration.experiment3_output_base_path + str(order) + "-grams.csv"
    print("Writing results to: " + out_file)
    write_csv_pitt(group_patient_ppl_dictionary, out_file)


def write_csv_pitt(group_patient_ppl_dictionary, eval_file):
    csv = "Group Id, Subject Id, Interview ID,P_Control, P_Dementia\n"

    # Save control group
    for partition in group_patient_ppl_dictionary:
        for patient in group_patient_ppl_dictionary[partition]:
            for interview in group_patient_ppl_dictionary[partition][patient]:
                csv += partition + "," + patient + "," + interview + "," + \
                       str(group_patient_ppl_dictionary[partition][patient][interview]["p_control"]) + "," + \
                       str(group_patient_ppl_dictionary[partition][patient][interview]["p_dementia"]) + "," + \
                       "\n"
    csv += "\n"

    with open(eval_file, 'w') as f:
        f.write(csv)

def build_vocabulary_file(vocabulary_fp):
    all_text = ""
    for category_file_name in os.listdir(configuration.experiment3_data_base_path + "ALL/"):
        if not category_file_name.startswith("."):
            category_file_path = configuration.experiment3_data_base_path + "ALL/" + category_file_name
            file_content = open(category_file_path, 'r').read().replace("\n\n","\n")
            all_text += file_content + "\n"

    open(vocabulary_fp, 'w').write(all_text)


if __name__ == '__main__':
    run_experiment(5)
