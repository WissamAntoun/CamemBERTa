# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions and classes related to optimization (weight updates)."""

import collections
import re

import tensorflow as tf

# import tensorflow_addons.optimizers as tfa_optimizers
from lamb import LAMB
from utils import log


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applys a warmup schedule on a given learning rate decay schedule."""

    def __init__(
        self,
        initial_learning_rate,
        decay_schedule_fn,
        warmup_steps,
        power=1.0,
        name=None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(
                warmup_percent_done, self.power
            )
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }


class ContantSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        return self.initial_learning_rate


def create_optimizer(
    init_lr,
    num_train_steps,
    num_warmup_steps,
    weight_decay_rate=0.01,
    layerwise_lr_decay=-1,
    n_transformer_layers=None,
    clip_norm=1.0,
    optimizer="adam",
    skip_adaptive=False,
    schedule="linear",
    power=1.0,
    beta_1=0.9,
    beta_2=0.999,
    end_lr=0.0,
):
    """Creates an optimizer with learning rate schedule."""
    # Implements linear decay of the learning rate.
    if schedule == "linear":
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=init_lr,
            decay_steps=num_train_steps - num_warmup_steps,
            end_learning_rate=end_lr,
            power=power,
        )
    if schedule == "constant":
        learning_rate_fn = ContantSchedule(init_lr)

    if num_warmup_steps:
        learning_rate_fn = WarmUp(
            initial_learning_rate=init_lr,
            decay_schedule_fn=learning_rate_fn,
            warmup_steps=num_warmup_steps,
        )
    layer_decay = None
    if layerwise_lr_decay > 0 and n_transformer_layers is not None:
        layer_decay = _get_layer_decay(layerwise_lr_decay, n_transformer_layers)

    if optimizer == "adam":
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate_fn,
            weight_decay=weight_decay_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=1e-6,
            clipnorm=clip_norm,
        )
        optimizer.exclude_from_weight_decay(
            var_names=["layer_norm", "bias", "LayerNorm"]
        )
    elif optimizer == "lamb":
        if skip_adaptive:
            skip_list = ["layer_norm", "bias", "LayerNorm"]
        else:
            skip_list = ["None"]
        log("Skip list for LAMB {}".format(skip_list))

        optimizer = LAMB(
            learning_rate=learning_rate_fn,
            weight_decay_rate=weight_decay_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=1e-6,
            exclude_from_weight_decay=["layer_norm", "bias", "LayerNorm"],
            exclude_from_layer_adaptation=skip_list,
        )
    else:
        raise ValueError("Unknown optimizer {}".format(optimizer))

    return optimizer


# Inspired from https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/optimizers/utils.py
class GradientAccumulator(object):
    """Distribution strategies-aware gradient accumulation utility."""

    def __init__(self):
        """Initializes the accumulator."""
        self._gradients = []
        self._accum_steps = tf.Variable(
            initial_value=0,
            dtype=tf.int64,
            trainable=False,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )

    @property
    def step(self):
        """Number of accumulated steps."""
        return self._accum_steps.value()

    @property
    def gradients(self):
        """The accumulated gradients."""
        return list(
            gradient.value() if gradient is not None else gradient
            for gradient in self._get_replica_gradients()
        )

    def __call__(self, gradients):
        """Accumulates :obj:`gradients`."""
        if not self._gradients:
            self._gradients.extend(
                [
                    (
                        tf.Variable(tf.zeros_like(gradient), trainable=False)
                        if gradient is not None
                        else gradient
                    )
                    for gradient in gradients
                ]
            )

        if len(gradients) != len(self._gradients):
            raise ValueError(
                "Expected %s gradients, but got %d"
                % (len(self._gradients), len(gradients))
            )

        for accum_gradient, gradient in zip(self._get_replica_gradients(), gradients):
            if accum_gradient is not None and gradient is not None:
                accum_gradient.assign_add(gradient)

        self._accum_steps.assign_add(1)

    def reset(self):
        """Resets the accumulated gradients."""
        if self._gradients:
            self._accum_steps.assign(0)

        for gradient in self._get_replica_gradients():
            if gradient is not None:
                gradient.assign(tf.zeros_like(gradient))

    def _get_replica_gradients(self):
        if tf.distribute.has_strategy():
            # In a replica context, we want to accumulate gradients on each replica
            # without synchronization, so we directly assign the value of the
            # current replica.
            replica_context = tf.distribute.get_replica_context()

            if (
                replica_context is None
                or tf.distribute.get_strategy().num_replicas_in_sync == 1
            ):
                return self._gradients

            return (
                gradient.device_map.select_for_current_replica(
                    gradient.values, replica_context
                )
                for gradient in self._gradients
                if gradient is not None
            )
        else:
            return self._gradients


# Inspired from https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/optimizers/utils.py
class GradientAccumulatorv2:
    def __init__(self):
        self._gradients = []
        self._accum_steps = None

    def zero(self, dtype):
        return tf.Variable(
            tf.constant(0, dtype=dtype),
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )

    @property
    def step(self):
        if self._accum_steps is None:
            self._accum_steps = self.zero(tf.int64)
        return self._accum_steps.value()

    @property
    def gradients(self):
        if not self._gradients:
            raise ValueError(
                "The accumulator should be called first to initialize the gradients"
            )
        return list(
            gradient.value() if gradient is not None else None
            for gradient in self._gradients
        )

    def reset(self):
        if not self._gradients:
            return
        self._accum_steps.assign(0)
        for gradient in self._gradients:
            if gradient is not None:
                gradient.assign(tf.zeros(tf.shape(gradient), dtype=gradient.dtype))

    def add_gradients(self, grads):
        if not self._gradients:
            _ = self.step
            self._gradients.extend(
                [
                    (
                        tf.Variable(
                            tf.zeros_like(g),
                            trainable=False,
                            synchronization=tf.VariableSynchronization.ON_READ,
                        )
                        if g is not None
                        else None
                    )
                    for g in grads
                ]
            )
        if len(grads) != len(self._gradients):
            raise ValueError(
                "Expected %s gradients, but got %d" % (len(self._gradients), len(grads))
            )

        for accum_grad, grad in zip(self._gradients, grads):
            if accum_grad is not None:
                accum_grad.assign_add(grad)

        self._accum_steps.assign_add(1)


class GradientAccumulatorv3:
    def __init__(self):
        self._gradients = []
        self._accum_steps = None

    def zero(self, dtype):
        return tf.Variable(
            tf.constant(0, dtype=dtype),
            trainable=False,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )

    @property
    def step(self):
        if self._accum_steps is None:
            self._accum_steps = self.zero(tf.int64)
        return self._accum_steps.value()

    @property
    def gradients(self):
        if not self._gradients:
            raise ValueError(
                "The accumulator should be called first to initialize the gradients"
            )
        return list(
            gradient.value() if gradient is not None else None
            for gradient in self._get_replica_gradients()
        )

    def reset(self):
        if not self._gradients:
            return
        self._accum_steps.assign(0)
        for gradient in self._gradients:
            if gradient is not None:
                gradient.assign(tf.zeros(tf.shape(gradient), dtype=gradient.dtype))

    def add_gradients(self, grads):
        if not self._gradients:
            _ = self.step
            self._gradients.extend(
                [
                    (
                        tf.Variable(tf.zeros_like(g), trainable=False)
                        if g is not None
                        else None
                    )
                    for g in grads
                ]
            )
        if len(grads) != len(self._gradients):
            raise ValueError(
                "Expected %s gradients, but got %d" % (len(self._gradients), len(grads))
            )

        for accum_grad, grad in zip(self._get_replica_gradients(), grads):
            if accum_grad is not None:
                accum_grad.assign_add(grad)

        self._accum_steps.assign_add(1)

    def _get_replica_gradients(self):
        if tf.distribute.has_strategy():
            # In a replica context, we want to accumulate gradients on each replica
            # without synchronization, so we directly assign the value of the
            # current replica.
            replica_context = tf.distribute.get_replica_context()

            if (
                replica_context is None
                or tf.distribute.get_strategy().num_replicas_in_sync == 1
            ):
                return self._gradients

            return (
                gradient.device_map.select_for_current_replica(
                    gradient.values, replica_context
                )
                for gradient in self._gradients
                if gradient is not None
            )
        else:
            return self._gradients


def _get_layer_decay(layer_decay, n_layers):
    """Have lower learning rates for layers closer to the input."""
    key_to_depths = collections.OrderedDict(
        {
            "/embeddings/": 0,
            "/embeddings_project/": 0,
            "/start_logits/": n_layers + 2,
            "/end_logits/": n_layers + 2,
            "/answer_class/": n_layers + 2,
            "/qa_outputs/": n_layers + 2,
        }
    )
    for layer in range(n_layers):
        key_to_depths["encoder/layer_._" + str(layer) + "/"] = layer + 1
    return {
        key: layer_decay ** (n_layers + 2 - depth)
        for key, depth in key_to_depths.items()
    }
