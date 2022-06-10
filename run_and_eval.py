import subprocess
import os

#
# File for fast running of training and eval
#

#train the net

trace = 200.0

tc_trace = trace
tc_trace_delay = trace

n_neurons = 400
inh = None

n_epochs = 4

wmin = None

gpu = True
delayed = False

filename = f'{int(trace)}ms_tc_trace_capped_no_delays'

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

train_and_full_eval += create_command("Auditory_network_loading_fixed_newandoldpopassignment.py", modelname=filename)

subprocess.run(train_and_full_eval, shell=True)

#run second time

trace = 1000.0

tc_trace = trace
tc_trace_delay = trace
n_epochs = 4

filename = f'{int(trace)}ms_tc_trace_capped_no_delays'

train_and_full_eval = ""

train_and_full_eval += create_command("Auditory_network_just_train.py")
train_and_full_eval += "; "

n_epochs = 1

train_and_full_eval += create_command("Auditory_network_loading_fixed_newandoldpopassignment.py", modelname=filename)

subprocess.run(train_and_full_eval, shell=True)

#run third time

trace = 5.0

tc_trace = trace
tc_trace_delay = trace
n_epochs = 4

delayed = True

filename = f'{int(trace)}ms_tc_trace_capped_with_delays'

train_and_full_eval = ""

train_and_full_eval += create_command("Auditory_network_just_train.py")
train_and_full_eval += "; "

n_epochs = 1

train_and_full_eval += create_command("Auditory_network_loading_fixed_newandoldpopassignment.py", modelname=filename)

subprocess.run(train_and_full_eval, shell=True)

#run fourth time

trace = 20.0

tc_trace = trace
tc_trace_delay = trace
n_epochs = 4

delayed = True

filename = f'{int(trace)}ms_tc_trace_capped_with_delays'

train_and_full_eval = ""

train_and_full_eval += create_command("Auditory_network_just_train.py")
train_and_full_eval += "; "

n_epochs = 1

train_and_full_eval += create_command("Auditory_network_loading_fixed_newandoldpopassignment.py", modelname=filename)

subprocess.run(train_and_full_eval, shell=True)