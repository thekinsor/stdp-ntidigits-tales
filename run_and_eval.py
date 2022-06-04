import subprocess
import os

#
# File for fast running of training and eval
#

#train the net

filename = "20ms_tc_trace_capped_400n_200_inh_1_epoch"
n_neurons = 400
tc_trace = 20.0
tc_trace_delay = 20.0
inh = 200.0

n_epochs = 4

gpu = True
delayed = True

def create_command(
        file_to_run: str,
        modelname: str = None,
    ):
    command = []

    command.append("python")

    command.append(file_to_run)

    command.append("--filename")
    command.append(filename)

    if gpu: command.append("--gpu")

    if delayed: command.append("--delayed")

    command.append("--n_epochs")
    command.append(f'{n_epochs}')

    if modelname is not None : 
        command.append("--modelname")
        command.append(modelname)

    command.append("--inh")
    command.append(f'{inh}')

    command.append("--n_neurons")
    command.append(f'{n_neurons}')

    command.append("--tc_trace")
    command.append(f'{tc_trace}')

    command.append("--tc_trace_delay")
    command.append(f'{tc_trace_delay}')

    str_command = ""

    for i in command:
        str_command += (i + " ")

    return str_command

# # Create directories
# directories = ["results", "results/" + filename, "results/" + filename + "/console_output"]
# for directory in directories:
#     if not os.path.exists(directory):
#         os.makedirs(directory, exist_ok=True)

# file_train = open(f'results/{filename}/console_output/training_script_output.txt', "w+")
# file_train_err = open(f'results/{filename}/console_output/training_script_err_output.txt', "w+")

train_and_full_eval = ""

train_and_full_eval += create_command("Auditory_network_just_train.py")
train_and_full_eval += "; "

train_and_full_eval += create_command("Auditory_network_loading_fixed_newandoldpopassignment.py", modelname=filename)
# train_and_full_eval += "; "

# train_and_full_eval += create_command("Auditory_network_test_loading_fixed_evalpop_four_different_schemes.py", modelname=filename)
# train_and_full_eval += "; "

# train_and_full_eval += create_command("Auditory_network_test_loading_fixed_evalpop_threshold_ballpark.py", modelname=filename)

# subprocess.run(create_command("Auditory_network_just_train.py") + "; " + create_command("Auditory_network_loading_fixed_newandoldpopassignment.py", modelname=filename),
#                 shell=True)

subprocess.run(train_and_full_eval, shell=True)

# file_train.close()
# file_train_err.close()

# file_eval = open(f'results/{filename}/console_output/eval_script_output.txt', "w+")
# file_eval_err = open(f'results/{filename}/console_output/eval_script_err_output.txt', "w+")

#subprocess.run(create_command("Auditory_network_loading_fixed_newandoldpopassignment.py", modelname= filename))#,
                #stdout=file_eval,
                #stderr= file_eval_err)

# file_eval.close()
# file_eval_err.close()


    
