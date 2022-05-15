import os
from turtle import delay
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from time import time as t
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import (
    all_activity,
    proportion_weighting,
    assign_labels,
)
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    # plot_assignments,
    plot_performance,
    plot_voltages,
)
from bindsnet.analysis.visualization import summary, plot_weights_movie
from modified_bindsnet_tales import SpikingNetwork, plot_confusion_matrix, plot_weights, plot_spikes_rate, \
    plot_input_spikes, plot_assignments
from NTIDIGITS import NTIDIGITS
from NMNIST import NMNIST, SparseToDense
from custom_modules_tales import FactoredDelayedConnection, DiehlAndCookNodesContinual, DiehlCookDelayedSTDP
from bindsnet.network.nodes import LIFNodes
from bindsnet.network.topology import Connection

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a SNN on the NTIDIGITS dataset.')
# Neuron parameters
parser.add_argument("--thresh", type=float, default=-52.0, help='Threshold for the membrane voltage.')
parser.add_argument("--tc_decay", type=float, default=215.0, help='Time constant for the membrane voltage decay.')
# Learning rule parameters
parser.add_argument("--x_tar", type=float, default=0.4, help='Target value for the pre-synaptic trace (STDP).')
# Network parameters
parser.add_argument("--n_neurons", type=int, default=100, help='Number of neurons in the excitatory layer.')
parser.add_argument("--exc", type=float, default=22.5, help='Strength of excitatory synapses.')
parser.add_argument("--inh", type=float, default=17.5, help='Strength of inhibitory synapses.')
parser.add_argument("--theta_plus", type=float, default=0.2, help='Step increase for the adaptive threshold.')
parser.add_argument("--som", dest="som", action="store_true", help='Enable for topological self-organisation.')
parser.add_argument("--recurrency", dest="recurrency", action="store_true", help='Enable for simple excitatory recurrent connections.')
parser.add_argument("--delayed", dest="delayed", action="store_true", help='Enable for delayed connections.')
# Data parameters
parser.add_argument("--n_test", type=int, default=None, help='Number of samples for the testing set (if None, '
                                                             'all are used)')
parser.add_argument("--n_train", type=int, default=None, help='Number of samples for the training set (if None, '
                                                              'all are used)')
parser.add_argument("--pattern_time", type=int, default=1000, help='Duration (in milliseconds) of a single pattern.')
parser.add_argument("--filename", type=str, default='test', help='Name for the experiment (and resulting files).')
# Simulation parameters
parser.add_argument("--dt", type=float, default=1.0, help='Simulation timestep.')
parser.add_argument("--n_epochs", type=int, default=1, help='Number of training epochs.')
parser.add_argument("--n_workers", type=int, default=-1, help='Number of parallel processes to be created.')
parser.add_argument("--seed", type=int, default=0, help='Seed for the pseudorandom number generation.')
parser.add_argument("--progress_interval", type=int, default=10, help='Frequency of training progress reports.')
parser.add_argument("--plot", dest="plot", action="store_true", help='Enable plotting (considerably slows the '
                                                                     'simulation).')
parser.add_argument("--gpu", dest="gpu", action="store_true", help='Enable GPU acceleration.')
# Defaults
parser.set_defaults(plot=False, gpu=False, som=False, recurrency=False, delayed=False)

args = parser.parse_args()

thresh = args.thresh
tc_decay = args.tc_decay
x_tar = args.x_tar
n_neurons = args.n_neurons
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
som = args.som
n_test = args.n_test
n_train = args.n_train
pattern_time = args.pattern_time
filename = args.filename
dt = args.dt
n_epochs = args.n_epochs
n_workers = args.n_workers
seed = args.seed
progress_interval = args.progress_interval
plot = args.plot
gpu = args.gpu
recurrency = args.recurrency
delayed = args.delayed

# Set up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False
np.random.seed(seed)

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

if n_workers == -1:
    n_workers = gpu * 4 * torch.cuda.device_count()

# Load training data
train_dataset = NTIDIGITS(root=os.path.join("./", "data"), download=True, train=True, dt=dt)

# Declare auxiliary variables and parameters
n_classes = 10
n_train = len(train_dataset) if n_train == None else n_train
update_interval = 5 #n_train // 600
data_dim = train_dataset.data[0].shape[1]
data_dim_sqrt = int(np.sqrt(data_dim))
n_neurons_sqrt = int(np.ceil(np.sqrt(n_neurons)))
c_inhib = torch.linspace(-5.0, -17.5, n_train // update_interval, device=device)
w_inhib = (torch.ones(n_neurons, n_neurons) - torch.diag(torch.ones(n_neurons))).to(device)
pattern_repetition_counter = 0

# Build the network
network = SpikingNetwork(n_neurons=n_neurons, inpt_shape=(1, data_dim), n_inpt=data_dim, dt=dt,
                         thresh=thresh, tc_decay=tc_decay, theta_plus=theta_plus, x_tar=x_tar,
                         weight_factor=1.0, exc=exc, inh=inh, som=som, start_inhib=-5.0, max_inhib=-17.5, recurrency=recurrency, delayed=delayed)
network_correct_data = torch.load("results/" + str(filename)+"/model_" + str(filename) + ".pt")

#create an additional exc layer with inhibiting competition
exc_layer_L2 = DiehlAndCookNodesContinual(
            n=n_neurons,  # PARAMETER
            traces=True, #has to be true to make recurrent connections
            rest=-65.0,
            reset=-60.0,
            thresh=thresh,  # PARAMETER
            refrac=5,
            tc_decay=tc_decay,  # PARAMETER
            tc_trace=20.0,
            tc_trace_delay = 20.0,
            theta_plus=theta_plus,  # PARAMETER
            tc_theta_decay= float(1e7),
        )
inh_layer_L2 = LIFNodes(n=n_neurons, traces=False, rest=-60.0, reset=-45.0, thresh=-40.0, tc_decay=10.0,
                             refrac=2, tc_trace=20.0)

#set up connections for the additional layers
w = 0.3 * torch.rand(n_neurons, n_neurons)

#random weights
d = torch.randint(low=0, high=200, size=(n_neurons, n_neurons))

#time constant for trace decays
alpha = np.exp(- dt / tc_decay)

exc1_exc2_conn = FactoredDelayedConnection(
    source=network.layers["Excitatory"],
    target=exc_layer_L2,
    w=w,
    update_rule=DiehlCookDelayedSTDP,  # FUTURE WORK: Other learning rules could also be explored.
    nu=float(5e-2),
    wmin=0.0,
    wmax=1.0,
    weight_factor=40.0,  # PARAMETER
    d=d,
    dmin = torch.min(d).int(),
    dmax = torch.max(d).int(),
    alpha=alpha,
    tc_trace_delay = 20.0,
    tc_trace=20.0,
    )
exc1_exc2_conn.update_rule.x_tar = x_tar  # PARAMETER

#exc2 to inh2 connection
w = exc * torch.diag(torch.ones(n_neurons))
exc_inh_conn_L2 = Connection(source=exc_layer_L2, target=inh_layer_L2, w=w, wmin=0, wmax=exc)

#and the connections back without som
w = -inh * (torch.ones(n_neurons, n_neurons) - torch.diag(torch.ones(n_neurons)))
inh_exc_conn_L2 = Connection(source=inh_layer_L2, target=exc_layer_L2, w=w, wmin=-inh, wmax=0)

network.add_layer(exc_layer_L2, name="Excitatory2")
network.add_layer(inh_layer_L2, name= "Inhibitory2")
network.add_connection(exc1_exc2_conn, source="Excitatory", target="Excitatory2")
network.add_connection(exc_inh_conn_L2, source="Excitatory2", target="Inhibitory2")
network.add_connection(inh_exc_conn_L2, source="Inhibitory2", target="Excitatory2")

network.load_state_dict(network_correct_data.state_dict())

network.connections['Excitatory', 'Excitatory2'].weight_factor = 40.0


if gpu:
    network.cuda(device="cuda")
print(summary(network))


# Record spikes during the simulation
excitatory_spikes = torch.tensor((int(pattern_time / dt), n_neurons), dtype=torch.bool, device=device)
spike_record = torch.zeros((update_interval, int(pattern_time / dt), n_neurons), device=device)
cumulative_spikes = torch.zeros((n_train // update_interval, n_neurons), device=device)

# Neuron assignments and spike proportions
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates
accuracy = {"all": [], "proportion": []}
proportion_pred = torch.zeros(update_interval, dtype=torch.int64, device=device)
label_tensor = torch.zeros(update_interval, dtype=torch.int64, device=device)
training_proportion_pred = torch.zeros(0, dtype=torch.int64, device=device)
training_label_tensor = torch.zeros(0, dtype=torch.int64, device=device)

# Set up monitors for spikes and voltages
exc_voltage_monitor = Monitor(
    network.layers["Excitatory"], ["v"], time=int(pattern_time / dt), device=device
)
inh_voltage_monitor = Monitor(
    network.layers["Inhibitory"], ["v"], time=int(pattern_time / dt), device=device
)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(pattern_time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"Input"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(pattern_time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None
cm_ax = None
cspikes_ax = None
input_s_im = None

# Load testing data
test_dataset = NTIDIGITS(root=os.path.join("./", "data"), download=True, train=False, dt=dt)
n_test = len(test_dataset) if n_test == None else n_test

# Sequence of accuracy estimates
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation
spike_record = torch.zeros((1, int(pattern_time / dt), n_neurons), device=device)
testing_proportion_pred = torch.zeros(n_test, dtype=torch.int64, device=device)
testing_label_tensor = torch.zeros(n_test, dtype=torch.int64, device=device)

data_dim = test_dataset.data[0].shape[1]
data_dim_sqrt = int(np.sqrt(data_dim))

# Test the network
print("\nBegin testing\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)
for step, batch in enumerate(test_dataset):
    if step >= n_test:
        break

    # Get next input sample
    inputs = {"Input": batch[0].view(int(pattern_time / dt), 1, 1, data_dim)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    network.run(inputs=inputs, time=pattern_time, input_time_dim=1)

    excitatory_spikes = spikes["Excitatory2"].get("s").squeeze()

    # Run the network on the input
    # network.connections['Input', 'Excitatory'].weight_factor = 1.2
    # for spikes_check in range(5):

    #     network.run(inputs=inputs, time=pattern_time, input_time_dim=1)

    #     excitatory_spikes = spikes["Excitatory"].get("s").squeeze()
    #     # If not enough spikes, present that sample again (with increased weight factor)
    #     if excitatory_spikes.sum().sum() < 2:
    #         network.connections['Input', 'Excitatory'].weight_factor *= 1.2
    #     else:
    #         break

    # Add to spikes recording
    spike_record[0].copy_(excitatory_spikes, non_blocking=True)

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch[1], device=device)

    # Get network predictions
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    testing_proportion_pred[step] = proportion_pred
    testing_label_tensor[step] = label_tensor

    network.reset_state_variables()  # Reset state variables
    pbar.set_description_str("Test progress: ")
    pbar.update()

plot_confusion_matrix(torch.Tensor.cpu(testing_proportion_pred), torch.Tensor.cpu(testing_label_tensor),
                      save=f'./results/{filename}/test_confusion_matrix_{filename}.png')

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Progress: %d / %d (%.4f seconds)" % (0 + 1, n_epochs, t() - start))
print("Testing complete.\n")

# Save logs.
with open(f'./results/{filename}/log_{filename}.txt', 'w+') as log_file:
    print(f'''
    All activity accuracy: {accuracy["all"] / n_test}
    Proportion weighting accuracy: {accuracy["proportion"] / n_test} \n
    Progress: {0 + 1} / {n_epochs} ({t() - start} seconds)
    Testing complete.\n
    Pattern repetitions during training: {pattern_repetition_counter}\n''', file=log_file)
