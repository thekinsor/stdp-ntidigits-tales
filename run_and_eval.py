import subprocess
import os

#
# Script for quickly and conveniently running several other training and eval scripts sequentially
#

#set parameters for the command arguments

trace = 20.0

tc_trace = trace
tc_trace_delay = trace

n_neurons = 400
inh = None

n_epochs = 1

wmin = None

gpu = True
delayed = True
capped = False
dlearning = False

recurrency = False

shutdown = True

filename = f'{int(trace)}ms_tc_trace_integrative_with_delays'

#create commandline command
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

    if recurrency: command.append("--recurrency")

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

#two layer stuff
n_neurons = 400
delayed = True
capped = True


trace = 5.0

tc_trace = trace
tc_trace_delay = trace

#construct the commands

filename = "LIF_0_200_400n_nodlearning_capped_trace_5ms_tc"

n_epochs = 1

train_and_full_eval += create_command("Auditory_network_loading_sdg_classifier.py", modelname=filename)

#if one wishes to shutdown after finishing
if(shutdown): train_and_full_eval += "; shutdown now"

#run it
subprocess.run(train_and_full_eval, shell=True)

