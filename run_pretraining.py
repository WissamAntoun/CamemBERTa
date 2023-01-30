# coding=utf-8
# Copyright 2020 The Google Research Authors.
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

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

"""Pre-trains a DebertaV3 model."""

import argparse
import json
import os
import time
from typing import Tuple

import horovod.tensorflow as hvd
import tensorflow as tf
from horovod.tensorflow.compression import Compression

import pretrain_utils
import utils
from configuration_deberta_v2 import DebertaV3PretrainingConfig
from configuration_roberta import RobertaPretrainingConfig
from model_training_utils import run_customized_training_loop
from modeling_tf_deberta_v2 import PretrainingModel as DebertaPretrainingModel
from modeling_tf_roberta import PretrainingModel as RobertaPretrainingModel
from official_utils.misc import distribution_utils
from optimization import create_optimizer
from utils import is_main_process, log, log_config, print_model_layers


def get_loss_fn(loss_factor=1.0):
    """Returns loss function for pretraining."""

    def _pretrain_loss_fn(losses, **unused_args):
        return tf.keras.backend.mean(losses) * loss_factor

    return _pretrain_loss_fn


def main():
    # Parse essential argumentss
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str)

    args = parser.parse_args()
    loaded_config = json.load(open(args.config_file))

    model_type = loaded_config["model_type"]
    if model_type == "deberta-v2":
        config = DebertaV3PretrainingConfig(**loaded_config)
        log("Using Deberta V2 model")
    elif model_type == "roberta":
        config = RobertaPretrainingConfig(**loaded_config)
        log("Using Roberta model")
    else:
        raise ValueError("Unknown model type: {}".format(model_type))

    tf.get_logger().setLevel("DEBUG")

    # Set up config cont'
    if config.load_weights and config.restore_checkpoint:
        raise ValueError(
            "`load_weights` and `restore_checkpoint` should not be on at the same time."
        )
    if config.phase2 and not config.restore_checkpoint:
        raise ValueError("`phase2` cannot be used without `restore_checkpoint`.")

    utils.heading("Config:")
    log_config(config)

    if config.use_horovod:
        # Set up tensorflow horovod
        hvd.init()

        log("Horovod Local Rank: %d" % hvd.local_rank(), all_rank=True)
        log("Horovod Rank: %d" % hvd.rank(), all_rank=True)
        from gpu_affinity import set_affinity

        set_affinity(hvd.local_rank())

    gpus = tf.config.experimental.list_physical_devices("GPU")
    log("Num GPUs Available: %d" % len(gpus), all_rank=True)
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if config.use_horovod:
            log("Setting Visible Gpu: %s" % str(gpus[hvd.local_rank()]), all_rank=True)
            tf.config.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy=config.distribution_strategy,
        all_reduce_alg=None,
        num_gpus=config.num_gpus,
        tpu_address=config.tpu_address,
    )

    if strategy:
        log("Number of cores used: ", strategy.num_replicas_in_sync)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    if config.amp:
        if config.distribution_strategy == "tpu":
            raise ValueError("TPU doesn't support AMP")
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Compute dtype: %s" % policy.compute_dtype)  # Compute dtype: float16
        print("Variable dtype: %s" % policy.variable_dtype)  # Variable dtype: float32

    tf.random.set_seed(config.seed)
    # tf.profiler.experimental.server.start(7789)

    # Save pretrain configs
    pretrain_config_json = os.path.join(config.checkpoints_dir, "pretrain_config.json")
    if is_main_process():
        utils.write_json(config.__dict__, pretrain_config_json)
        log("Configuration saved in {}".format(pretrain_config_json))

    def _get_model() -> Tuple[
        DebertaPretrainingModel, RobertaPretrainingModel, tf.keras.optimizers.Optimizer
    ]:
        # Set up model
        if model_type == "deberta-v2":
            model = DebertaPretrainingModel(config)
        elif model_type == "roberta":
            model = RobertaPretrainingModel(config)
        else:
            raise ValueError("Unknown model type: {}".format(model_type))

        print_model_layers(model)
        # Set up optimizer
        optimizer = create_optimizer(
            init_lr=config.learning_rate,
            num_train_steps=config.num_train_steps,
            num_warmup_steps=config.num_warmup_steps,
            weight_decay_rate=config.weight_decay_rate,
            optimizer=config.optimizer,
            skip_adaptive=config.skip_adaptive,
            power=config.lr_decay_power,
            beta_1=config.opt_beta_1,
            beta_2=config.opt_beta_2,
            end_lr=config.end_lr,
        )

        if config.amp:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                optimizer, dynamic=True
            )

        return model, optimizer

    dataset = pretrain_utils.get_pretrain_dataset_fn(config, config.train_batch_size)

    metrics = dict()
    metrics["train_perf"] = tf.keras.metrics.Mean(name="train_perf")
    metrics["learning_rate"] = tf.keras.metrics.Mean(name="learning_rate")
    metrics["loss_scale"] = tf.keras.metrics.Mean(name="loss_scale")
    metrics["total_loss"] = tf.keras.metrics.Mean(name="total_loss")
    metrics["masked_lm_accuracy"] = tf.keras.metrics.Accuracy(name="masked_lm_accuracy")
    metrics["masked_lm_loss"] = tf.keras.metrics.Mean(name="masked_lm_loss")
    if config.electra_objective:
        metrics["sampled_masked_lm_accuracy"] = tf.keras.metrics.Accuracy(
            name="sampled_masked_lm_accuracy"
        )
        if config.disc_weight > 0:
            metrics["disc_loss"] = tf.keras.metrics.Mean(name="disc_loss")
            metrics["disc_auc"] = tf.keras.metrics.AUC(name="disc_auc")
            metrics["disc_accuracy"] = tf.keras.metrics.Accuracy(name="disc_accuracy")
            metrics["disc_precision"] = tf.keras.metrics.Accuracy(name="disc_precision")
            metrics["disc_recall"] = tf.keras.metrics.Accuracy(name="disc_recall")

    trained_model = run_customized_training_loop(
        strategy=strategy,
        config=config,
        model_fn=_get_model,
        loss_fn=get_loss_fn(
            loss_factor=1.0 / strategy.num_replicas_in_sync
            if config.scale_loss and strategy
            else 1.0
        ),
        train_input_fn=dataset,
        metric_fn_dict=metrics,
        hvd=hvd if config.use_horovod else None,
        run_eagerly=config.debug,
    )

    return config


if __name__ == "__main__":
    start_time = time.time()
    config = main()
    log("Total Time:{:.4f}".format(time.time() - start_time))
    log("Finished")
