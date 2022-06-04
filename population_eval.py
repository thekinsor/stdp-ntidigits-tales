from cProfile import label
from typing import Optional, Tuple
import torch


def pop_prediction(
    spikes: torch.Tensor, label_profiles: torch.Tensor, n_labels: int
) -> torch.Tensor:
    # language=rst
    """
    Classify data with the label with highest aligned output activity profile.

    :param spikes: Binary tensor of shape ``(n_samples, time, n_neurons)`` of a layer's
        spiking activity.
    :param label_profiles: A vector of shape ``(n_neurons, n_labels)`` of neuron label assignments.
    :param n_labels: The number of target labels in the data.
    :return: Predictions tensor of shape ``(n_samples,)`` resulting from the "all
        activity" classification scheme.
    """

    #TODO implement

    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)

    #create normalized activity profile
    activity_profile = spikes #torch.nn.functional.normalize(spikes,dim=1,p=2)

    #calculate align of each profile to the label profiles
    fits = torch.mm(activity_profile, label_profiles)

    # Predictions are arg-max of population activity vector products with label_profiles.
    return torch.sort(fits, dim=1, descending=True)[1][:, 0]

def pop_decorrelated_prediction(
    spikes: torch.Tensor, label_profiles: torch.Tensor, n_labels: int, zca_matrix: torch.Tensor
) -> torch.Tensor:
    # language=rst
    """
    Classify data with the label with highest aligned output activity profile with whitening.

    :param spikes: Binary tensor of shape ``(n_samples, time, n_neurons)`` of a layer's
        spiking activity.
    :param label_profiles: A vector of shape ``(n_neurons, n_labels)`` of neuron label assignments.
    :param n_labels: The number of target labels in the data.
    :return: Predictions tensor of shape ``(n_samples,)`` resulting from the "all
        activity" classification scheme.
    """

    #TODO implement

    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)

    #create decorrelated activity profile
    activity_profile = torch.mm(spikes, zca_matrix) #torch.nn.functional.normalize(spikes,dim=1,p=2)

    #calculate align of each profile to the label profiles
    fits = torch.mm(activity_profile, label_profiles)

    # Predictions are arg-max of population activity vector products with label_profiles.
    return torch.sort(fits, dim=1, descending=True)[1][:, 0]

def pop_avg_prediction(
    spikes: torch.Tensor, label_profiles: torch.Tensor, n_labels: int, avg_activity: torch.Tensor, zca_matrix: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # language=rst
    """
    Classify data with the label with highest aligned output activity profile while using both spikes and profiles with average diff.

    :param spikes: Binary tensor of shape ``(n_samples, time, n_neurons)`` of a layer's
        spiking activity.
    :param label_profiles: A vector of shape ``(n_neurons, n_labels)`` of activity profiles with averaged diff.
    :param n_labels: The number of target labels in the data.
    :return: Predictions tensor of shape ``(n_samples,)`` resulting from the "all
        activity" classification scheme.
    """

    #TODO implement

    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)

    #create normalized activity profile
    activity_profile = spikes/spikes.sum(dim=1, keepdim=True)

    if(zca_matrix is not None):
        activity_profile = torch.mm(activity_profile, zca_matrix)

    #create the avg 
    activity_profile -= avg_activity.t()

    #norm it
    #norm = torch.abs(activity_profile)
    #activity_profile /= norm.sum()

    #calculate align of each profile to the label profiles
    fits = torch.mm(activity_profile, label_profiles)

    # Predictions are arg-max of population activity vector products with label_profiles.
    return torch.sort(fits, dim=1, descending=True)[1][:, 0]

def pop_diff_prediction(
    spikes: torch.Tensor, label_profiles: torch.Tensor, n_labels: int
) -> torch.Tensor:
    # language=rst
    """
    Classify data with the label with lowest output activity mismatch.

    :param spikes: Binary tensor of shape ``(n_samples, time, n_neurons)`` of a layer's
        spiking activity.
    :param label_profiles: A vector of shape ``(n_neurons, n_labels)`` of neuron label assignments.
    :param n_labels: The number of target labels in the data.
    :return: Predictions tensor of shape ``(n_samples,)`` resulting from the "all
        activity" classification scheme.
    """

    #TODO implement

    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)

    n_neurons = label_profiles.shape[0]
    n_samples = spikes.shape[0]

    #create normalized activity profile
    activity_profile = spikes/spikes.sum(dim=1, keepdim=True)

    #calculate difference between each profile and the label profiles (abs value error)
    preprocess = activity_profile.view((n_samples,n_neurons,1))

    fits = torch.abs(label_profiles-preprocess)

    #sum over all neurons in one population in each sample for each label
    fits = fits.sum(dim=1)

    # Predictions are arg-max of population activity vector products with label_profiles.
    return torch.sort(fits, dim=1, descending=False)[1][:, 0]


def assign_pop_labels(
    spikes: torch.Tensor,
    labels: torch.Tensor,
    n_labels: int,
    label_profiles: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    n_train: int = 2239,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # language=rst
    """
    Assign labels to the neurons based on highest average spiking activity.

    :param spikes: Binary tensor of shape ``(n_samples, time, n_neurons)`` of a single
        layer's spiking activity.
    :param labels: Vector of shape ``(n_samples,)`` with data labels corresponding to
        spiking activity.
    :param n_labels: The number of target labels in the data.
    :param label_profiles: If passed, these represent spike label_profiles from a previous
        ``assign_labels()`` call.
    :param alpha: Rate of decay of label assignments.
    :return: Tuple of class assignments, per-class spike proportions, and per-class
        firing label_profiles.
    """
    n_neurons = spikes.size(2)

    if label_profiles is None:
        label_profiles = torch.zeros((n_neurons, n_labels), device=spikes.device)

    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)

    #create normalized activity profile
    activity_profile = spikes/spikes.sum(dim=1, keepdim=True)

    #TODO test different "influences". For now we get the average activity profile per label for n_train samples going over the training data set
    #activity_profile /= n_train

    for i in range(n_labels):
        # Count the number of samples with this label.
        n_labeled = torch.sum(labels == i).float()

        if n_labeled > 0:
            # Get indices of samples with this label.
            indices = torch.nonzero(labels == i).view(-1)

            # Compute average firing label_profiles for this label.
            label_profiles[:, i] = alpha * label_profiles[:, i] + (torch.sum(activity_profile[indices], 0))
        
    #normalize label profiles but don't divide through zero
    # thing_to_normalize = label_profiles.sum(dim=0,keepdim=True)
    # label_profiles = torch.where((thing_to_normalize != 0), label_profiles/thing_to_normalize, torch.zeros_like(label_profiles))
    # label_profiles[label_profiles != label_profiles] = 0  # Set NaNs to 0

    return label_profiles