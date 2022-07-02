from turtle import delay
from typing import Optional, Union, Tuple, List, Sequence, Iterable, Type, Dict, Sized
import numpy as np
import torch
from matplotlib import pyplot as plt, ticker
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import euclidean
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.nn import Parameter
from torch.nn.modules.utils import _pair
import torch.nn as nn
from torchvision import models
from bindsnet.learning import PostPre, WeightDependentPostPre, LearningRule
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, DiehlAndCookNodes, AdaptiveLIFNodes, Nodes
from bindsnet.network.topology import Connection, LocalConnection, AbstractConnection, Conv2dConnection


#FILE WITH MODIFIED/CUSTOM NODES, CONNECTIONS AND LEARNING RULES
#BY TALES BRAIG


class DiehlAndCookNodesContinual(Nodes):
    # language=rst
    """
    Layer of leaky integrate-and-fire (LIF) neurons with adaptive thresholds (modified for Diehl & Cook 2015
    replication). In this implementation the homeostasis is kept even in eval mode.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        tc_trace_delay: Union[float, torch.Tensor] = 200.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = -52.0,
        rest: Union[float, torch.Tensor] = -65.0,
        reset: Union[float, torch.Tensor] = -65.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        theta_plus: Union[float, torch.Tensor] = 0.05,
        tc_theta_decay: Union[float, torch.Tensor] = 1e7,
        lbound: float = None,
        one_spike: bool = True,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of Diehl & Cook 2015 neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param theta_plus: Voltage increase of threshold after spiking.
        :param tc_theta_decay: Time constant of adaptive threshold decay.
        :param lbound: Lower bound of the voltage.
        :param one_spike: Whether to allow only one spike per timestep.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        self.register_buffer("rest", torch.tensor(rest))  # Rest voltage.
        self.register_buffer("reset", torch.tensor(reset))  # Post-spike reset voltage.
        self.register_buffer("thresh", torch.tensor(thresh))  # Spike threshold voltage.
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )  # Post-spike refractory period.
        self.register_buffer(
            "tc_decay", torch.tensor(tc_decay)
        )  # Time constant of neuron voltage decay.
        self.register_buffer(
            "tc_trace", torch.tensor(tc_trace)
        )  # Time constant of trace decay.
        self.register_buffer(
            "tc_trace_delay", torch.tensor(tc_trace_delay)
        )  # Time constant of trace decay for delays.
        self.register_buffer(
            "decay", torch.empty_like(self.tc_decay)
        )  # Set in compute_decays.
        self.register_buffer(
            "theta_plus", torch.tensor(theta_plus)
        )  # Constant threshold increase on spike.
        self.register_buffer(
            "tc_theta_decay", torch.tensor(tc_theta_decay)
        )  # Time constant of adaptive threshold decay.
        self.register_buffer(
            "theta_decay", torch.empty_like(self.tc_theta_decay)
        )  # Set in compute_decays.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer("theta", torch.zeros(*self.shape))  # Adaptive thresholds.
        self.register_buffer("x", torch.zeros(*self.shape))  # Traces
        self.register_buffer(
            "refrac_count", torch.FloatTensor()
        )  # Refractory period counters.
        

        self.lbound = lbound  # Lower bound of voltage.
        self.one_spike = one_spike  # One spike per timestep.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        if self.traces:
            # Decay and set spike traces for delay learning.
            self.x *= self.trace_decay_delay
            #print("trace: post" + str(self.x.sum()))

            if self.traces_additive:
                self.x += self.trace_scale * self.s.float()
            else:
                self.x.masked_fill_(self.s.bool(), 1)

        # Decay voltages and adaptive thresholds.
        self.v = self.decay * (self.v - self.rest) + self.rest
        self.theta *= self.theta_decay

        # Integrate inputs.
        self.v += (self.refrac_count <= 0).float() * x

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        # Check for spiking neurons.
        self.s = self.v >= self.thresh + self.theta

        # Refractoriness, voltage reset, and adaptive thresholds.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        self.theta += self.theta_plus * self.s.float().sum(0)

        # Choose only a single neuron to spike.
        if self.one_spike:
            if self.s.any():
                _any = self.s.view(self.batch_size, -1).any(1)
                ind = torch.multinomial(
                    self.s.float().view(self.batch_size, -1)[_any], 1
                )
                _any = _any.nonzero()
                self.s.zero_()
                self.s.view(self.batch_size, -1)[_any, ind] = 1

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.
        self.x.zero_()

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).
        self.theta_decay = torch.exp(
            -self.dt / self.tc_theta_decay
        )  # Adaptive threshold decay (per timestep).
        self.trace_decay = torch.exp(
            -self.dt / self.tc_trace
        )  # Trace decay (per timestep).
        self.trace_decay_delay = torch.exp(
            -self.dt / self.tc_trace_delay
        ) # Trace decay (per timestep) for delay learning.

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)

class FactoredDelayedConnection(Connection):
    def __init__(
            self,
            source: Nodes,
            target: Nodes,
            nu: Optional[Union[float, Sequence[float]]] = None,
            reduction: Optional[callable] = None,
            weight_decay: float = 0.0,
            weight_factor: float = 1.0,
            alpha: float = np.exp(-1/20.),
            alpha_delay: float = np.exp(-1/20.),
            tc_trace_delay: float = 20.0,
            tc_trace: float = 20.0,
            dlearning: bool = False,
            **kwargs
    ) -> None:
        # language=rst
        """
        Instantiates a :code:`Connection` object, slightly modified to include a factor multiplying the weights to
        promote post-synaptic spikes as well as synaptic delays.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        :param weight_factor: Factor for the weights.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor d: Value of delays.
        :param torch.Tensor b: Target population bias.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        :param int dmin: Minimum allowed value on the connection delays.
        :param int dmax: Maximum allowed value on the connection delays.
        :param float alpha: Trace decay factor
        """
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)
        self.weight_factor = weight_factor

        #set up delay values
        self.d = kwargs.get('d', torch.zeros_like(self.w))
        self.dmin = kwargs.get('dmin', None)
        self.dmax = kwargs.get('dmax', None)
        self.capped = kwargs.get('capped', False)

        #trace decay value (for now equal for integrative and capped)
        self.alpha = alpha
        self.alpha_delay = alpha_delay
        self.tc_trace = tc_trace
        self.tc_trace_delay = tc_trace_delay
        self.dlearning = dlearning

        #print(dlearning)

        assert self.d is not None, "Delays have to be set for Delayed connections"
        assert self.dmax is not None or self.dmin is not None, "Delay constraints have to be set"

        #add one for processing
        self.d.add_(1)

        #record presynaptic traces 
        self.xpre = torch.zeros_like(self.w)
        self.xpre_capped = torch.zeros_like(self.w)
        #self.xpost = torch.zeros_like(self.w)

        #set up delay memory
        self.delayed_spikes = torch.zeros_like(self.d)

        self.counter=0

        #put everything correctly so it is part of network graph
        self.d = torch.nn.Parameter(self.d, requires_grad=False)
        self.delayed_spikes = torch.nn.Parameter(self.delayed_spikes, requires_grad=False)
        self.xpre = torch.nn.Parameter(self.xpre, requires_grad=False)
        self.xpre_capped = torch.nn.Parameter(self.xpre_capped, requires_grad=False)
        #self.xpost = torch.nn.Parameter(self.xpost, requires_grad=False)


    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using connection weights and weight factor.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
                decaying spike activation).
        """

        #decay traces
        # print(self.xpre.sum())
        self.xpre *= self.alpha
        self.xpre_capped *= self.alpha_delay
        #self.xpost *= self.alpha
        # print(self.xpre.sum())

        #save current spikes to mapped delayed spikes
        delays_to_save = (self.d*s.view(-1, s.size(0)).int()).int()
        self.delayed_spikes += delays_to_save

        # print(s.view(-1, s.size(0)).int())

        self.delayed_spikes = torch.nn.Parameter(self.delayed_spikes.abs(), requires_grad=False)
        
        #current spikes to process at each synapse
        current_spikes = torch.where(self.delayed_spikes % self.d == 1, 1, 0)

        # print(self.delayed_spikes)
        # print(delays_to_save)

        # Compute multiplication of spike activations by weights and add bias delayed by the synaptic delays.
        if self.b is None:
            post = ((self.w * self.weight_factor)*current_spikes.float()).sum(dim=0)
        else:
            post = ((self.w * self.weight_factor)*current_spikes.float()).sum(dim=0) + self.b

        #save the new presynaptic trace (capped trace)
        self.xpre.add_(current_spikes.float())
        self.xpre_capped.masked_fill_(current_spikes.bool(), 1)
        
        #decrease stored spikes counter
        self.delayed_spikes.sub_(1)
        self.delayed_spikes = torch.nn.Parameter(torch.clamp(self.delayed_spikes, min=0), requires_grad=False)

        # print(self.delayed_spikes)

        # print("s: " + str(s.sum()))
        # print("self.delayed_spikes: " + str(self.delayed_spikes.sum()))
        # print("self.xpre: " + str(self.xpre.sum()))
        # print("post: " + str(post.sum()))
        # print("delays_to_save: " + str(delays_to_save.sum()))
        # print("self.current_spikes: " + str(current_spikes.sum()) + "\n")

        # if self.counter == 10: 4/0

        self.counter+=1

        return post.view(s.size(0), *self.target.shape)
        

    def reset_state_variables(self):
        super().reset_state_variables

        self.xpre *= 0
        self.xpre_capped *= 0
        self.delayed_spikes *= 0

class DiehlCookDelayedSTDP(LearningRule):
    # language=rst
    """
    STDP rule from `(Diehl & Cook 2015)<https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_,
    triggered only by post-synaptic spiking activity with inclusion of delays. The post-synaptic update is positive and is dependent on the
    magnitude of the synaptic weights and on the pre-synaptic trace.
    Delays are learned from pre and post synaptic traces
    """

    def __init__(
            self,
            connection: AbstractConnection,
            nu: Optional[float] = None,
            reduction: Optional[callable] = None,
            weight_decay: float = 0.0,
            **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``DiehlCookSTDP`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``WeightDependentPostPre`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        assert self.source.traces, "Pre-synaptic nodes must record spike traces."
        assert (
                connection.wmin != -np.inf and connection.wmax != np.inf
        ), "Connection must define finite wmin and wmax."

        self.wmin = connection.wmin
        self.wmax = connection.wmax
        self.x_tar = 0.4

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Diehl&Cook's learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        source_x = torch.zeros_like(self.connection.xpre)

        #use intended trace
        if self.connection.capped: source_x = self.connection.xpre_capped
        else: source_x = self.connection.xpre

        target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float()

        update = 0
        update_steps = 0

        # Post-synaptic update.
        if self.nu is not None:
            outer_product = self.reduction(torch.where(target_s.bool(), source_x - self.x_tar,
                                                       torch.zeros_like(source_x)), dim=0)
            update += self.nu[1] * outer_product * torch.pow((self.wmax - self.connection.w), 0.2)
        
            if(self.connection.dlearning):

                #capped traces
                source_x_capped = self.connection.xpre_capped


                target_x_capped = torch.ones_like(source_x_capped)

                if(self.connection.target.traces_additive): target_x_capped *= self.connection.target.x.clamp(min=0.,max=1.)
                else: target_x_capped *= self.connection.target.x

                #print("Target x capped: " + str(target_x_capped.sum()))

                #-dt/tc_trace
                #source_alpha_ln = -self.connection.source.dt/self.connection.tc_trace_delay
                #target_alpha_ln = -self.connection.target.dt/self.connection.target.tc_trace_delay

                #source_alpha = self.connection.alpha_delay
                #target_alpha = self.connection.target.trace_decay_delay

                #delay updates correct post and pre
                #decrease_delay_update = torch.where((source_x_capped > source_alpha) & (target_x_capped > 0) & (target_x_capped <= target_alpha),
                #                             torch.round(torch.log(target_x_capped)/target_alpha_ln), torch.zeros_like(target_x_capped))

                #increase_delay_update = torch.where((target_x_capped > target_alpha) & (source_x_capped > 0) & (source_x_capped <= 1),
                #                             torch.round(torch.log(source_x_capped)/source_alpha_ln), torch.zeros_like(source_x_capped))

                # print("DEcreases: " + str(((source_x_capped > source_alpha).sum())))# & (target_x_capped > 0) & (target_x_capped <= target_alpha)).sum()))
                # print("Increases: " + str(((target_x_capped.max()))) + str(((target_x_capped > target_alpha) & (source_x_capped > 0) & (source_x_capped <= 1)).sum()))
                # print("\n")

                # #some factors to try out TODO
                #delay_lr = float(0.01)
                #homeostasis_factor = float(1/2.0)
                #sigmoid_factor = float(5.0)
                delay_pre_tar = float(0.8)
                #delay_post_tar = float(-0.8)

                # #delay update soft (if matched spikes, no update, if too early delay, else deacrease)
                # steps = delay_steps*torch.where((source_x > 0.) & (source_x < 1.), torch.ceil((1 - source_x)), source_x - 1)

                #simple rule, if pre trace bigger than certain value, maybe another one can still come after to build a burst so add some delay
                #if trace already decayed more than a certain value, go back to learn the burst
                #if there was no pre spike for the out spike then up the delay to, maybe later there is a correspondence to learn still
                update_value = torch.ones_like(self.connection.d)
                update_steps = torch.where((source_x_capped > delay_pre_tar) | (source_x_capped == 0), -update_value, update_value)
                update_steps = self.reduction(torch.where((target_s.bool()) & (source_x_capped < delay_pre_tar),
                                                            update_steps, torch.zeros_like(update_steps)), dim=0)

                # #delay update hard (if matched spikes, no update, if too early delay, else deacrease)
                # #steps = torch.where((source_x > 0.) & (source_x < 1.), np.log(1./self.connection.alpha)*torch.log(source_x), delay_steps*(source_x - 1))

                #demyelinization = homeostasis_factor*torch.ones_like(source_x_capped)

                #delay update soft (if they fire together input fires even a bit earlier)
                #steps = delay_lr*self.connection.dmax *torch.where(source_x_capped > delay_x_tar, torch.exp(-source_x_capped), demyelinization)

                #post pre approach
                #first calculate difference between xpost and xpre * sigmoid factor and whereever they are over the targets record the sigmoids
                #xdiff = sigmoid_factor*(target_x_capped - source_x_capped)
                #delay_update = delay_lr*self.connection.dmax*torch.where((xdiff >= delay_post_tar) & (xdiff <= delay_pre_tar),
                #                                                                        torch.sigmoid(xdiff), demyelinization)
            

                #delay_update = delay_lr*self.connection.dmax*self.reduction(torch.where((target_s.bool()),
                #                                                -torch.exp(-source_x_capped), demyelinization), dim=0)

                #print(((target_s.bool()) & (source_x_capped > delay_x_tar)).sum())

                #add some homeostatic normalization and demyelinization
                #delay_update /= delay_update.sum(dim=0)
                #delay_update *= delay_lr*(self.connection.dmax - self.connection.dmin)
                #delay_update.add_(homeostasis_steps*self.connection.dmax)

                #print(steps)

                #delay_update.round()
                #print(delay_update.sum())

        self.connection.w += update
        #decrease_delay_update = torch.where(decrease_delay_update.int() <= self.connection.d,
        #                decrease_delay_update, torch.zeros_like(decrease_delay_update)) #don't decrease if there wouldn't be an impact anyway
        # self.connection.d -= decrease_delay_update.int()
        # self.connection.d += increase_delay_update.int()

        if(self.connection.dlearning):
            #save old delays
            old_delays = self.connection.d

            self.connection.d += update_steps.int()
            self.connection.d.clamp_(min=self.connection.dmin + 1, max=self.connection.dmax + 1)

            #update saved delays to not create errors of still propagating delayed spikes
            self.connection.delayed_spikes.add_((self.connection.delayed_spikes // old_delays) * (old_delays - self.connection.d))

        #homeostasis rule includes decrease and increase in one
        #self.connection.d += delay_update.int()
        #self.connection.d.clamp_(min=self.connection.dmin + 1, max=self.connection.dmax + 1)

        #update saved delays to not create errors of still propagating delayed spikes
        #self.connection.delayed_spikes.add_((self.connection.delayed_spikes // old_delays) * (old_delays - self.connection.d))

        #print(self.connection.d.max())

        #print(self.connection.d)


        super().update()

class PostPreDelayed(LearningRule):
    # language=rst
    """
    Simple delayed STDP rule involving both pre- and post-synaptic spiking activity. By default,
    pre-synaptic update is negative and the post-synaptic update is positive.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``PostPre`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``PostPre`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        assert (
            self.source.traces and self.target.traces
        ), "Both pre- and post-synaptic nodes must record spike traces."

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        # Pre-synaptic update.
        if self.nu[0]:
            source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
            target_x = self.target.x.view(batch_size, -1).unsqueeze(1) * self.nu[0]
            self.connection.w -= self.reduction(torch.bmm(source_s, target_x), dim=0)
            del source_s, target_x

        # Post-synaptic update.
        if self.nu[1]:
            target_s = (
                self.target.s.view(batch_size, -1).unsqueeze(1).float() * self.nu[1]
            )
            source_x = self.connection.xpre_capped
            self.connection.w += self.reduction(source_x*target_s, dim=0)
            del source_x, target_s

        super().update()