from datetime import datetime
import os

# --------------------------------------------------------------------------#
#                                 Logging                                   #
# --------------------------------------------------------------------------#
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
directory_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# -------------------------------------- #
#               Experiment 1             #
# -------------------------------------- #
experiment1_data_base_path = "../resources/data/experiment1/input/"
experiment1_train_data_base_path = "../resources/data/experiment1/datasets/"
experiment1_models_base_path = "../resources/models/experiment1/"
experiment1_output_base_path = "../resources/data/experiment1/output/"

# -----      GPT-2      ----- #
experiment1_gpt2_model_name = "gpt2"
experiment1_gpt2_batch_size = 16
experiment1_gpt2_arguments_file = "../resources/models/experiment1/args.json"
experiment1_gpt2_models_base_path = experiment1_models_base_path + "gpt2/"

# -----     N-grams     ----- #


# -------------------------------------- #
#               Experiment 2             #
# -------------------------------------- #
experiment2_data_base_path = "../resources/data/experiment2/input/"
experiment2_speeches_base_path = experiment2_data_base_path + "speeches/"
experiment2_train_data_base_path = experiment2_data_base_path + "train_sets/"
experiment2_test_data_base_path = experiment2_speeches_base_path + "datasets/"
experiment2_models_base_path = "../resources/models/experiment2/"
experiment2_output_base_path = "../resources/data/experiment2/output/"

# -----      GPT-2      ----- #
experiment2_gpt2_model_name = "gpt2"
experiment2_gpt2_batch_size = 16
experiment2_gpt2_arguments_file = "../resources/models/experiment2/args.json"
experiment2_gpt2_models_base_path = experiment2_models_base_path + "gpt2/"

# -----     N-grams     ----- #


# -------------------------------------- #
#               Experiment 3             #
# -------------------------------------- #
experiment3_data_base_path = "../resources/data/experiment3/input/pitt/"
experiment3_train_data_base_path = "../resources/data/experiment3/datasets/"
experiment3_models_base_path = "../resources/models/experiment3/"
experiment3_output_base_path = "../resources/data/experiment3/output/"

# -----      GPT-2      ----- #
experiment3_gpt2_model_name = "gpt2"
experiment3_gpt2_batch_size = 16
experiment3_gpt2_arguments_file = "../resources/models/experiment3/args.json"
experiment3_gpt2_models_base_path = experiment3_models_base_path + "gpt2/"

# -----     N-grams     ----- #