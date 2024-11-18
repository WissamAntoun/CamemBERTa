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
""" Roberta and Roberta for Deberta style pretraining model configuration"""

# Modified by Wissam Antoun - Almanach - Inria Paris 2024

import os

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "roberta-base": "https://huggingface.co/roberta-base/resolve/main/config.json",
    "roberta-large": "https://huggingface.co/roberta-large/resolve/main/config.json",
    "roberta-large-mnli": "https://huggingface.co/roberta-large-mnli/resolve/main/config.json",
    "distilroberta-base": "https://huggingface.co/distilroberta-base/resolve/main/config.json",
    "roberta-base-openai-detector": "https://huggingface.co/roberta-base-openai-detector/resolve/main/config.json",
    "roberta-large-openai-detector": "https://huggingface.co/roberta-large-openai-detector/resolve/main/config.json",
}


class RobertaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RobertaModel`] or a [`TFRobertaModel`]. It is
    used to instantiate a RoBERTa model according to the specified arguments, defining the model architecture.


    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    The [`RobertaConfig`] class directly inherits [`BertConfig`]. It reuses the same defaults. Please check the parent
    class for more information.

    Examples:

    ```python
    >>> from transformers import RobertaConfig, RobertaModel

    >>> # Initializing a RoBERTa configuration
    >>> configuration = RobertaConfig()

    >>> # Initializing a model from the configuration
    >>> model = RobertaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    Arguments:
        vocab_size (`int`, *optional*, defaults to 128100):
            Vocabulary size of the RObertamodel. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`RobertaModel`].
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
            The vocabulary size of the `token_type_ids` passed when calling [`RobertaModel`] or [`TFRobertaModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-7):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            The value used to pad input_ids.
        position_biased_input (`bool`, *optional*, defaults to `False`):
            Whether add absolute position embedding to content embedding.
        layer_norm_eps (`float`, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
    """

    model_type = "roberta"

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
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-7,
        pad_token_id=0,
        position_biased_input=True,
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
        self.max_position_embeddings = max_position_embeddings  # This needs to be equal to actual max position embeddings plus padding_token_id plus 1
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.position_biased_input = position_biased_input
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps


class RobertaPretrainingConfig(object):
    """Defines extra pre-training hyperparamters"""

    def __init__(self, model_name="", **kwargs):
        # super().__init__(**kwargs)
        self.model_name = model_name
        self.seed = 42

        self.debug = False  # debug mode for quickly running things
        self.do_train = True
        self.do_eval = False  # evaluate generator/discriminator on unlabeled data
        self.phase2 = False
        self.record_gradients = False
        self.profile = False

        # amp
        self.distribution_strategy = "one_device"
        self.use_horovod = False
        self.num_gpus = 1
        self.tpu_address = ""
        self.amp = False
        self.xla = True
        self.fp16_compression = False
        self.bf16 = False

        # optimizer type
        self.optimizer = "adam"
        self.gradient_accumulation_steps = 1
        self.lr_schedule = "linear"

        # lamb whitelisting for LN and biases
        self.skip_adaptive = False

        # loss functions
        self.electra_objective = True  # if False, use the BERT objective instead\
        self.model_type = "roberta"  # 'bert' or 'debertav2' or 'roberta'
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
        self.type_vocab_size = 1
        self.position_biased_input = True
        self.data_prep_working_dir = os.getenv("DATA_PREP_WORKING_DIR", "")
        self.repeat_dataset = True
        self.update(kwargs)
        # default locations of data files

        self.pretrain_tfrecords = os.path.join(
            "data", "pretrain_tfrecords/pretrain_data.tfrecord*"
        )
        self.vocab_file = os.path.join("vocab", "vocab.txt")
        self.ignore_ids_dict = {
            "[PAD]": 0,
            "[CLS]": 1,
            "[SEP]": 2,
            "[UNK]": 3,
            "[MASK]": 4,
        }
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
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
