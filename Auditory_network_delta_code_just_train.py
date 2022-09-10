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
from modified_bindsnet_tales_delta_code import SpikingNetwork, plot_confusion_matrix, plot_weights, plot_spikes_rate, \
    plot_input_spikes, plot_assignments
from NTIDIGITS import NTIDIGITS
from NMNIST import NMNIST, SparseToDense

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a SNN on the NTIDIGITS dataset.')
# Neuron parameters
parser.add_argument("--thresh", type=float, default=-52.0, help='Threshold for the membrane voltage.')
parser.add_argument("--tc_decay", type=float, default=215.0, help='Time constant for the membrane voltage decay.')
parser.add_argument("--burst_time", type=int, default=20, help='Tolerance time between spikes for "continuous" activation.')
# Learning rule parameters
parser.add_argument("--x_tar", type=float, default=0.4, help='Target value for the pre-synaptic trace (STDP).')
parser.add_argument("--tc_trace", type=float, default=20.0, help='Time constant for trace decay.')
parser.add_argument("--tc_trace_delay", type=float, default=5.0, help='Time constant for capped trace decay.')
# Network parameters
parser.add_argument("--n_neurons", type=int, default=100, help='Number of neurons in the excitatory layer.')
parser.add_argument("--wmin", type=float, default=0.0, help='Minimum allowed weights.')
parser.add_argument("--exc", type=float, default=22.5, help='Strength of excitatory synapses.')
parser.add_argument("--inh", type=float, default=17.5, help='Strength of inhibitory synapses.')
parser.add_argument("--theta_plus", type=float, default=0.2, help='Step increase for the adaptive threshold.')
parser.add_argument("--som", dest="som", action="store_true", help='Enable for topological self-organisation.')
parser.add_argument("--recurrency", dest="recurrency", action="store_true", help='Enable for simple excitatory recurrent connections.')
parser.add_argument("--delayed", dest="delayed", action="store_true", help='Enable for delayed connections.')
parser.add_argument("--dlearning", dest="dlearning", action="store_true", help='Enable for learning delayed connections.')
parser.add_argument("--capped", dest="capped", action="store_true", help='Use capped traces for learning.')
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
parser.set_defaults(plot=False, gpu=False, som=False, recurrency=False, delayed=False, capped=False, dlearning=False)

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
tc_trace = args.tc_trace
tc_trace_delay = args.tc_trace_delay
wmin = args.wmin
capped = args.capped
dlearning = args.dlearning
burst_time = args.burst_time

# Create directories
directories = ["results",
                "results/" + filename,
                "results/" + filename + "/weights_images",
                "results/" + filename + "/assignment_images/",
                "results/" + filename + "/weights_images/on_weights",
                "results/" + filename + "/weights_images/off_weights"]
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

if(delayed):
    if not os.path.exists("results/" + filename + "/delay_images/"):
        os.makedirs("results/" + filename + "/delay_images/", exist_ok=True)
    if not os.path.exists("results/" + filename + "/delay_images/on_delays/"):
        os.makedirs("results/" + filename + "/delay_images/on_delays/", exist_ok=True)
    if not os.path.exists("results/" + filename + "/delay_images/off_delays/"):
        os.makedirs("results/" + filename + "/delay_images/off_delays/", exist_ok=True)

if(recurrency):
    if not os.path.exists("results/" + filename + "/rec_weights_images/"):
        os.makedirs("results/" + filename + "/rec_weights_images/", exist_ok=True)

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
update_interval = n_train // 60
data_dim = train_dataset.data[0].shape[1]
data_dim_sqrt = int(np.sqrt(data_dim))
n_neurons_sqrt = int(np.ceil(np.sqrt(n_neurons)))
c_inhib = torch.linspace(-5.0, -17.5, n_train // update_interval, device=device)
w_inhib = (torch.ones(n_neurons, n_neurons) - torch.diag(torch.ones(n_neurons))).to(device)
pattern_repetition_counter = 0

# Build the network
network = SpikingNetwork(n_neurons=n_neurons, inpt_shape=(1, data_dim), n_inpt=data_dim, dt=dt,
                         thresh=thresh, tc_decay=tc_decay, theta_plus=theta_plus, x_tar=x_tar,
                         weight_factor=40.0, exc=exc, inh=inh, som=som, start_inhib=-5.0, max_inhib=-17.5,
                         recurrency=recurrency, delayed=delayed, tc_trace=tc_trace, tc_trace_delay=tc_trace_delay, wmin=wmin,
                         capped=capped, dlearning=dlearning, burst_time=burst_time)
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
for layer in set(network.layers) - {"Input", "ONCells", "OFFCells"}:
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

# Train the network
print("\nBegin training.\n")
start = t()
for epoch in range(n_epochs):
    labels = []
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(dataloader)):
        if step > n_train:
            break
        # Growing inhibition strategy
        elif network.som and (step > 0 and step % update_interval == 0):
            for i in range(network.n_neurons):
                for j in range(network.n_neurons):
                    if i != j:
                        x1, y1 = i // network.n_sqrt, i % network.n_sqrt
                        x2, y2 = j // network.n_sqrt, j % network.n_sqrt
        
                        w_inhib[i, j] = max(network.max_inhib, c_inhib[(step - 1) // update_interval] *
                                            network.inhib_scaling * np.sqrt(euclidean([x1, y1], [x2, y2])))
            network.connections['Inhibitory', 'Excitatory'].w.copy_(w_inhib)
        # Two-level inhibition strategy
        # elif network.som and step == 0.1 * n_train:
        #     w = -network.inh * (torch.ones(network.n_neurons, network.n_neurons)
        #                         - torch.diag(torch.ones(network.n_neurons)))
        #     network.connections['Inhibitory', 'Excitatory'].w.copy_(w_inhib)

        # Get next input sample
        inputs = {"Input": batch[0].view(int(pattern_time / dt), 1, 1, data_dim)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Progress updates
        if step % update_interval == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels, device=device)

            # Get network predictions
            all_activity_pred = all_activity(
                spikes=spike_record,
                assignments=assignments,
                n_labels=n_classes,
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )

            # Compute network accuracy according to available classification strategies
            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
            )
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred).item()
                / len(label_tensor)
            )
            tqdm.write(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            tqdm.write(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                " (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )
            # Assign labels to excitatory layer neurons
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )
            # FUTURE WORK: Limit the amount of labels used for the neuron assignments, and compare how the
            # performance of the network varies against the % of labels used.

            training_proportion_pred = torch.cat((training_proportion_pred, proportion_pred), dim=0)
            training_label_tensor = torch.cat((training_label_tensor, label_tensor), dim=0)
            labels = []

        labels.append(batch[1])

        # Run the network on the input
        #network.connections['Input', 'Excitatory'].weight_factor = 1.0
        network.run(inputs=inputs, time=pattern_time, input_time_dim=1)

        # Get voltage recording
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        excitatory_spikes = spikes["Excitatory"].get("s").squeeze()


        #pattern repetition mechanism
        # for spikes_check in range(5):
        #     network.run(inputs=inputs, time=pattern_time, input_time_dim=1)

        #     # Get voltage recording
        #     exc_voltages = exc_voltage_monitor.get("v")
        #     inh_voltages = inh_voltage_monitor.get("v")

        #     # If not enough spikes, present that sample again (with an increased weight factor)
        #     excitatory_spikes = spikes["Excitatory"].get("s").squeeze()
        #     if excitatory_spikes.sum().sum() < 2:
        #         network.connections['Input', 'Excitatory'].weight_factor *= 1.2
        #         pattern_repetition_counter += 1
        #     else:
        #         break

        # Add to spikes recording
        spike_record[step % update_interval].copy_(excitatory_spikes, non_blocking=True)

        # Optionally plot simulation information
        if plot:
            on_exc_weights = network.connections[("ONCells", "Excitatory")].w
            off_exc_weights = network.connections[("OFFCells", "Excitatory")].w
            square_weights_on = get_square_weights(on_exc_weights.view(data_dim, n_neurons), n_neurons_sqrt,
                                                data_dim_sqrt)
            square_weights_off = get_square_weights(off_exc_weights.view(data_dim, n_neurons), n_neurons_sqrt,
                                                data_dim_sqrt)
            square_assignments = get_square_assignments(assignments, n_neurons_sqrt)
            spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
            voltages = {"Excitatory": exc_voltages, "Inhibitory": inh_voltages}
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            weights_im_on = plot_weights(square_weights_on, im=weights_im_on)
            weights_im_off = plot_weights(square_weights_off, im=weights_im_off)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(accuracy, x_scale=update_interval, ax=perf_ax)
            voltage_ims, voltage_axes = plot_voltages(voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line")

            plt.pause(1e-8)

        # Save plots at checkpoints and at the end of the training
        if (step >= (n_train - 1)) or (step % update_interval == 0 and step > 0):
            cumulative_spikes[min(n_train // update_interval - 1, (step - 1) // update_interval), :].add_(torch.sum(spike_record.long(), (0, 1)))
            on_exc_weights = network.connections[("ONCells", "Excitatory")].w
            off_exc_weights = network.connections[("OFFCells", "Excitatory")].w
            square_weights_on = get_square_weights(on_exc_weights.view(data_dim, n_neurons), n_neurons_sqrt,
                                                data_dim_sqrt)
            square_weights_off = get_square_weights(off_exc_weights.view(data_dim, n_neurons), n_neurons_sqrt,
                                                data_dim_sqrt)
            if(delayed):
                on_exc_delays = network.connections[("ONCells", "Excitatory")].d
                off_exc_delays = network.connections[("OFFCells", "Excitatory")].d
                dmax_on = on_exc_delays.max()
                dmin_on = on_exc_delays.min()
                dmax_off = off_exc_delays.max()
                dmin_off = off_exc_delays.min()
                square_delays_on = get_square_weights(on_exc_delays.view(data_dim, n_neurons), n_neurons_sqrt,
                                                data_dim_sqrt)
                square_delays_off = get_square_weights(off_exc_delays.view(data_dim, n_neurons), n_neurons_sqrt,
                                                data_dim_sqrt)
                plot_weights(square_delays_on, wmin=dmin_on, wmax=dmax_on,
                            save=f'./results/{filename}/delay_images/on_delays/delays_{filename}_{step}_epoch_{epoch}.png')
                plot_weights(square_delays_off, wmin=dmin_off, wmax=dmax_off,
                            save=f'./results/{filename}/delay_images/off_delays/delays_{filename}_{step}_epoch_{epoch}.png')
            if(recurrency):
                input_rec_weights = network.connections[("Excitatory", "Excitatory")].w
                square_rec_weights = get_square_weights(input_rec_weights.view(n_neurons, n_neurons), n_neurons_sqrt,
                                                n_neurons_sqrt)
                plot_weights(square_rec_weights,
                            save=f'./results/{filename}/rec_weights_images/weights_{filename}_{step}_epoch_{epoch}.png')

            square_assignments = get_square_assignments(assignments, n_neurons_sqrt)
            plot_weights(square_weights_on, im=weights_im,
                         save=f'./results/{filename}/weights_images/on_weights/weights_{filename}_{step}_epoch_{epoch}.png', wmin=wmin)
            plot_weights(square_weights_off, im=weights_im,
                         save=f'./results/{filename}/weights_images/off_weights/weights_{filename}_{step}_epoch_{epoch}.png', wmin=wmin)
            plot_assignments(square_assignments, im=assigns_im, save=f'./results/{filename}/assignment_images/assignments'
                                                                     f'_{filename}_{step}_epoch_{epoch}.png')
            plot_performance(accuracy, x_scale=update_interval, ax=perf_ax,
                             save=f'./results/{filename}/performance_train_{filename}.png')
            plot_confusion_matrix(torch.Tensor.cpu(training_proportion_pred), torch.Tensor.cpu(training_label_tensor),
                                  save=f'./results/{filename}/confusion_matrix_train_{filename}_epoch_{epoch}.png')
            plot_spikes_rate(cumulative_spikes, save=f'./results/{filename}/cumulative_spikes_train_{filename}_epoch_{epoch}.png',
                             update_interval=update_interval)
            torch.save(network, f'./results/{filename}/model_{filename}.pt')
            network.save(f'./results/{filename}/network_{filename}.npz')

        network.reset_state_variables()  # Reset state variables

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")