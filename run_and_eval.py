import subprocess
import os

#
# File for fast running of training and eval
#

#train the net

trace = 20.0

tc_trace = trace
tc_trace_delay = trace

n_neurons = 400
inh = None

n_epochs = 4

wmin = None

gpu = True
delayed = True
capped = False
dlearning = True

shutdown = True

filename = f'{int(trace)}ms_tc_trace_integrative_with_delay_learning_dtar08_inverted_rdstpd_dlearning_but_decrease_at0trace'

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

    if capped: command.append("--capped")

    if dlearning: command.append("--dlearning")

    command.append("--n_epochs")
    command.append(f'{n_epochs}')

    if modelname is not None : 
        command.append("--modelname")
        command.append(modelname)

    if inh is not None:
        command.append("--inh")
        command.append(f'{inh}')
    
    if wmin is not None:
        command.append("--wmin")
        command.append(f'{wmin}')

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

train_and_full_eval = ""

train_and_full_eval += create_command("Auditory_network_just_train.py")
train_and_full_eval += "; "

n_epochs = 1
dlearning = False

train_and_full_eval += create_command("Auditory_network_loading_fixed_newandoldpopassignment.py", modelname=filename)

#print(train_and_full_eval)

subprocess.run(train_and_full_eval, shell=True)


if(shutdown): subprocess.run("shutdown now", shell=True)