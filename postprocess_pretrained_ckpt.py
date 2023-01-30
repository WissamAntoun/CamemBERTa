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

# Modified by Wissam Antoun - Almanach - Inria Paris 2022/2023

import argparse
import collections
import json
import os
import shutil

import numpy as np
import tensorflow as tf

from configuration_deberta_v2 import DebertaV3PretrainingConfig
from configuration_roberta import RobertaPretrainingConfig
from modeling_tf_deberta_v2 import PretrainingModel as DebertaPretrainingModel
from modeling_tf_roberta import PretrainingModel as RobertaPretrainingModel
from utils import heading, log, log_config, print_model_layers

# doesn't work because of nvidia logger :) so i just copy the tokenizer XD
# from fast_tokenizer.tokenization_deberta_v2_fast import DebertaV2TokenizerFast


def from_pretrained_ckpt(args):

    loaded_config = json.load(open(args.config_file))

    # Set up model
    model_type = loaded_config["model_type"]
    if model_type == "deberta-v2":
        config = DebertaV3PretrainingConfig(**loaded_config)
        log("Using Deberta V2 model")
    elif model_type == "roberta":
        config = RobertaPretrainingConfig(**loaded_config)
        log("Using Roberta model")
    else:
        raise ValueError("Unknown model type: {}".format(model_type))

    heading("Config:")
    log_config(config)

    if config.amp:
        policy = tf.keras.mixed_precision.experimental.Policy(
            "mixed_float16", loss_scale="dynamic"
        )
        tf.keras.mixed_precision.experimental.set_policy(policy)
        print("Compute dtype: %s" % policy.compute_dtype)  # Compute dtype: float16
        print("Variable dtype: %s" % policy.variable_dtype)  # Variable dtype: float32

    # Set up model
    if model_type == "deberta-v2":
        model = DebertaPretrainingModel(config)
    elif model_type == "roberta":
        model = RobertaPretrainingModel(config)
    else:
        raise ValueError("Unknown model type: {}".format(model_type))

    print_model_layers(model)
    # Load checkpoint
    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(0),
        phase2=tf.Variable(True),
        model=model,
    )

    manager = tf.train.CheckpointManager(
        checkpoint, config.checkpoints_dir, max_to_keep=config.keep_checkpoint_max
    )

    pretrained_ckpt_path = (
        args.pretrained_checkpoint
        if args.pretrained_checkpoint and manager.latest_checkpoint is not None
        else manager.latest_checkpoint
    )

    checkpoint.restore(pretrained_ckpt_path).expect_partial()
    log(
        " ** Restored from {} at step {}".format(
            pretrained_ckpt_path, int(checkpoint.step)
        )
    )

    output_dir = (
        args.output_dir
        if args.output_dir
        else os.path.join(config.checkpoints_dir, "postprocessed")
    )
    log(" ** Output dir: {}".format(output_dir))

    if config.electra_objective:
        log(" ** Model was trained using ELECTRA objective")
        disc_dir = os.path.join(output_dir, "discriminator")
        gen_dir = os.path.join(output_dir, "generator")
    else:
        log(" ** Model was trained using MLM objective")
        gen_dir = output_dir

    heading(" ** Saving generator")
    model.generator(model.generator.dummy_inputs)
    model.generator.update_embeddings()
    model.generator.save_pretrained(gen_dir)
    heading(" ** Saving Tokenizer")
    fast_tokenizer_path = os.path.join(config.vocab_file, "fast_tokenizer")
    log(f"Tokenizer files are from {fast_tokenizer_path}")
    for files in os.listdir(fast_tokenizer_path):
        log(f"Copying {files}")
        shutil.copy(os.path.join(fast_tokenizer_path, files), gen_dir)

    if config.electra_objective:
        heading(" ** Saving discriminator")
        if model_type == "deberta-v2" and "debertav2" in model.discriminator.__dict__:
            discriminator = model.discriminator.debertav2
            discriminator(discriminator.dummy_inputs)

            if config.shared_embeddings:
                assert np.all(
                    discriminator.get_input_embeddings().weight
                    == model.generator.get_input_embeddings().weight
                )

            discriminator.update_embeddings()
            print_model_layers(discriminator)
            discriminator.save_pretrained(disc_dir)
        else:
            model.discriminator(model.discriminator.dummy_inputs)
            if config.shared_embeddings:
                assert np.all(
                    model.discriminator.get_input_embeddings().weight
                    == model.generator.get_input_embeddings().weight
                )

            model.discriminator.update_embeddings()
            print_model_layers(model.discriminator)
            model.discriminator.save_pretrained(disc_dir)

        heading(" ** Saving Tokenizer")
        fast_tokenizer_path = os.path.join(config.vocab_file, "fast_tokenizer")
        log(f"Tokenizer files are from {fast_tokenizer_path}")
        for files in os.listdir(fast_tokenizer_path):
            log(f"Copying {files}")
            shutil.copy(os.path.join(fast_tokenizer_path, files), disc_dir)


if __name__ == "__main__":
    # Parse essential args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--pretrained_checkpoint")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    from_pretrained_ckpt(args)
