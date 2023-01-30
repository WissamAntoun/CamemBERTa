# coding=utf-8
# Copyright 2020, Microsoft and the HuggingFace Inc. team.
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
""" DeBERTa-v2 and DeBERTaV3 pretraining model configuration"""

# Modified by Wissam Antoun - Almanach - Inria Paris 2022/2023

import os

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/deberta-v2-xlarge": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/config.json",
    "microsoft/deberta-v2-xxlarge": "https://huggingface.co/microsoft/deberta-v2-xxlarge/resolve/main/config.json",
    "microsoft/deberta-v2-xlarge-mnli": "https://huggingface.co/microsoft/deberta-v2-xlarge-mnli/resolve/main/config.json",
    "microsoft/deberta-v2-xxlarge-mnli": "https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli/resolve/main/config.json",
}


class DebertaV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DebertaV2Model`]. It is used to instantiate a
    DeBERTa-v2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the DeBERTa
    [microsoft/deberta-v2-xlarge](https://huggingface.co/microsoft/deberta-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Arguments:
        vocab_size (`int`, *optional*, defaults to 128100):
            Vocabulary size of the DeBERTa-v2 model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`DebertaV2Model`].
        hidden_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 24):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"`, `"gelu"`, `"tanh"`, `"gelu_fast"`, `"mish"`, `"linear"`, `"sigmoid"` and `"gelu_new"`
            are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 0):
            The vocabulary size of the `token_type_ids` passed when calling [`DebertaModel`] or [`TFDebertaModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-7):
            The epsilon used by the layer normalization layers.
        relative_attention (`bool`, *optional*, defaults to `True`):
            Whether use relative position encoding.
        max_relative_positions (`int`, *optional*, defaults to -1):
            The range of relative positions `[-max_position_embeddings, max_position_embeddings]`. Use the same value
            as `max_position_embeddings`.
        pad_token_id (`int`, *optional*, defaults to 0):
            The value used to pad input_ids.
        position_biased_input (`bool`, *optional*, defaults to `False`):
            Whether add absolute position embedding to content embedding.
        pos_att_type (`List[str]`, *optional*):
            The type of relative position attention, it can be a combination of `["p2c", "c2p"]`, e.g. `["p2c"]`,
            `["p2c", "c2p"]`, `["p2c", "c2p"]`.
        layer_norm_eps (`float`, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
    """
    model_type = "deberta-v2"

    def __init__(
        self,
        vocab_size=128100,
        hidden_size=1536,
        embedding_size=1536,
        num_hidden_layers=24,
        num_attention_heads=24,
        intermediate_size=6144,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=0,
        initializer_range=0.02,
        layer_norm_eps=1e-7,
        conv_kernel_size=3,
        conv_act="gelu",
        relative_attention=True,
        position_buckets=256,
        norm_rel_ebd="layer_norm",
        max_relative_positions=-1,
        pad_token_id=0,
        position_biased_input=False,
        share_att_key=True,
        pos_att_type="p2c|c2p",
        pooler_dropout=0,
        pooler_hidden_act="gelu",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.relative_attention = relative_attention
        self.position_buckets = position_buckets
        self.norm_rel_ebd = norm_rel_ebd
        self.max_relative_positions = max_relative_positions
        self.pad_token_id = pad_token_id
        self.position_biased_input = position_biased_input
        self.share_att_key = share_att_key

        # Backwards compatibility
        if type(pos_att_type) == str:
            pos_att_type = [x.strip() for x in pos_att_type.lower().split("|")]

        self.pos_att_type = pos_att_type
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.conv_kernel_size = conv_kernel_size
        self.conv_act = conv_act

        self.pooler_hidden_size = kwargs.get("pooler_hidden_size", hidden_size)
        self.pooler_dropout = pooler_dropout
        self.pooler_hidden_act = pooler_hidden_act


class DebertaV3PretrainingConfig(object):
    """Defines extra pre-training hyperparamters"""

    def __init__(self, model_name="", **kwargs):
        # super().__init__(**kwargs)
        self.model_name = model_name
        self.seed = 42

        self.debug = False  # debug mode for quickly running things
        self.do_train = True  # pre-train DeBERTa
        self.do_eval = False  # evaluate generator/discriminator on unlabeled data
        self.phase2 = False

        # amp
        self.distribution_strategy = "one_device"
        self.use_horovod = False
        self.num_gpus = 1
        self.tpu_address = ""
        self.amp = True
        self.xla = True
        self.fp16_compression = False

        # optimizer type
        self.optimizer = "adam"
        self.gradient_accumulation_steps = 1

        # lamb whitelisting for LN and biases
        self.skip_adaptive = False

        # loss functions
        self.electra_objective = True  # if False, use the BERT objective instead
        self.model_type = "deberta-v2"  # 'bert' or 'debertav2' or 'roberta'
        self.gen_weight = 1.0  # masked language modeling / generator loss
        self.disc_weight = 50.0  # discriminator loss
        self.mask_prob = 0.15  # percent of input tokens to mask out / replace

        # optimization
        self.learning_rate = 5e-4
        self.lr_decay_power = 0.5
        self.weight_decay_rate = 0.01
        self.num_warmup_steps = 10000
        self.opt_beta_1 = 0.878
        self.opt_beta_2 = 0.974
        self.end_lr = 0.0
        self.scale_loss = False

        # training settings
        self.log_freq = 10
        self.skip_checkpoint = False
        self.save_checkpoints_steps = 1000
        self.eval_every_n_steps = 1000
        self.num_train_steps = 1000000
        self.num_eval_steps = 100
        self.keep_checkpoint_max = 5  # maximum number of recent checkpoint files to keep;  change to 0 or None to keep all checkpoints
        self.restore_checkpoint = None
        self.load_weights = False

        # model settings
        self.model_size = "base"  # one of "small", "base", or "large"
        # override the default transformer hparams for the provided model size; see
        # modeling.BertConfig for the possible hparams and util.training_utils for
        # the defaults
        self.model_hparam_overrides = (
            kwargs["model_hparam_overrides"]
            if "model_hparam_overrides" in kwargs
            else {}
        )
        self.vocab_size = (
            32001  # number of tokens in the vocabulary should be 32001 for camembert
        )
        self.do_lower_case = True  # lowercase the input?

        # generator settings
        self.uniform_generator = False  # generator is uniform at random
        self.shared_embeddings = True  # share generator/discriminator token embeddings?
        self.disentangled_gradients = False  # use disentangled gradients for discriminator?  # self.untied_generator = True  # tie all generator/discriminator weights?
        self.generator_layers = 1.0  # frac of discriminator layers for generator
        self.generator_hidden_size = 0.25  # frac of discrim hidden size for gen
        self.disallow_correct = False  # force the generator to sample incorrect
        # tokens (so 15% of tokens are always
        # fake)
        self.temperature = 1.0  # temperature for sampling from generator

        # batch sizes
        self.max_seq_length = 128
        self.train_batch_size = 128
        self.eval_batch_size = 128

        self.results_dir = "results"
        self.json_summary = None

        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 0
        self.relative_attention = True
        self.position_buckets = 256
        self.position_biased_input = False
        self.data_prep_working_dir = os.getenv("DATA_PREP_WORKING_DIR", "")
        self.update(kwargs)
        # default locations of data files

        self.pretrain_tfrecords = os.path.join(
            "data", "pretrain_tfrecords/pretrain_data.tfrecord*"
        )
        self.vocab_file = os.path.join("vocab", "vocab.txt")
        self.ignore_ids_dict = {"[SEP]": 2, "[CLS]": 1, "[MASK]": 32000}
        self.model_dir = os.path.join(self.results_dir, "models", model_name)
        self.checkpoints_dir = os.path.join(self.model_dir, "checkpoints")
        self.weights_dir = os.path.join(self.model_dir, "weights")
        self.results_txt = os.path.join(self.model_dir, "unsup_results.txt")
        self.results_pkl = os.path.join(self.model_dir, "unsup_results.pkl")
        self.log_dir = os.path.join(self.model_dir, "logs")

        self.max_predictions_per_seq = int(
            (self.mask_prob + 0.005) * self.max_seq_length
        )

        # defaults for different-sized model
        if self.model_size == "base":
            self.hidden_size = 768
            self.embedding_size = 768
            self.num_hidden_layers = 12
            self.intermediate_size = 3072
            if self.hidden_size % 64 != 0:
                raise ValueError(
                    "Hidden size {} should be divisible by 64. Number of attention heads is hidden size {} / 64 ".format(
                        self.hidden_size, self.hidden_size
                    )
                )
            self.num_attention_heads = int(self.hidden_size / 64.0)
        elif self.model_size == "large":
            self.hidden_size = 1024
            self.embedding_size = 1024
            self.num_hidden_layers = 24
            self.intermediate_size = 4096
            if self.hidden_size % 64 != 0:
                raise ValueError(
                    "Hidden size {} should be divisible by 64. Number of attention heads is hidden size {} / 64 ".format(
                        self.hidden_size, self.hidden_size
                    )
                )
            self.num_attention_heads = int(self.hidden_size / 64.0)
        elif self.model_size == "xsmall":
            self.hidden_size = 384
            self.embedding_size = 384
            self.num_hidden_layers = 12
            self.intermediate_size = 2536
            if self.hidden_size % 64 != 0:
                raise ValueError(
                    "Hidden size {} should be divisible by 64. Number of attention heads is hidden size {} / 64 ".format(
                        self.hidden_size, self.hidden_size
                    )
                )
            self.num_attention_heads = int(self.hidden_size / 64.0)
        elif self.model_size == "small":
            self.hidden_size = 768
            self.embedding_size = 768
            self.num_hidden_layers = 6
            self.intermediate_size = 3072
            if self.hidden_size % 64 != 0:
                raise ValueError(
                    "Hidden size {} should be divisible by 64. Number of attention heads is hidden size {} / 64 ".format(
                        self.hidden_size, self.hidden_size
                    )
                )
            self.num_attention_heads = int(self.hidden_size / 64.0)
        else:
            raise ValueError(
                "--model_size : 'xsmall', 'small', 'base' and 'large' supported only."
            )
        self.update(kwargs)

        if self.tpu_address == "colab":
            self.tpu_address = "grpc://" + os.environ["XRT_TPU_CONFIG"].split(";")[2]
            print("Using COLAB TPU address:", self.tpu_address)

        if self.vocab_size % 8 != 0:
            self.vocab_size += 8 - (self.vocab_size % 8)

    def update(self, kwargs):
        for k, v in kwargs.items():
            if v is not None:
                self.__dict__[k] = v
