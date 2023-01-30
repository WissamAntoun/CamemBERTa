# coding=utf-8
# Copyright 2021 Microsoft and The HuggingFace Inc. team. All rights reserved.
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
""" TF 2.0 DeBERTa-v2 model with DeBERTaV3 GDES pretraining capabilities."""

# Modified by Wissam Antoun - Almanach - Inria Paris 2022/2023

import collections
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

import pretrain_utils
from activations_tf import get_tf_activation
from configuration_deberta_v2 import (
    DebertaV2Config,
    DebertaV3PretrainingConfig,
    PretrainedConfig,
)
from modeling_tf_outputs import (
    TFBaseModelOutput,
    TFMaskedLMOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    unpack_inputs,
)
from tf_utils import shape_list
from utils import heading, log, log_config

logger = logging.getLogger(__name__)

tf.get_logger().setLevel("DEBUG")
logging2 = tf.get_logger()
logging2.propagate = False

_CONFIG_FOR_DOC = "DebertaV2Config"
_TOKENIZER_FOR_DOC = "DebertaV2Tokenizer"
_CHECKPOINT_FOR_DOC = "kamalkraj/deberta-v2-xlarge"

TF_DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "kamalkraj/deberta-v2-xlarge",
    # See all DeBERTa models at https://huggingface.co/models?filter=deberta-v2
]

tnp.experimental_enable_numpy_behavior()

# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaContextPooler with Deberta->DebertaV2
class TFDebertaV2ContextPooler(tf.keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.pooler_hidden_size, name="dense")
        self.dropout = TFDebertaV2StableDropout(
            config.pooler_dropout,
            dtype=tf.float16 if config.amp else tf.float32,
            name="dropout",
        )
        self.config = config

    def call(self, hidden_states, training: bool = False):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token, training=training)
        pooled_output = self.dense(context_token)
        pooled_output = get_tf_activation(self.config.pooler_hidden_act)(pooled_output)
        return pooled_output

    @property
    def output_dim(self) -> int:
        return self.config.hidden_size


# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaXSoftmax with Deberta->DebertaV2
class TFDebertaV2XSoftmax(tf.keras.layers.Layer):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (`tf.Tensor`): The input tensor that will apply softmax.
        mask (`tf.Tensor`): The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax
    """

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs: tf.Tensor, mask: tf.Tensor):

        rmask = tf.logical_not(tf.cast(mask, tf.bool))
        output = tf.where(rmask, tf.constant(float("-inf"), inputs.dtype), inputs)
        output = tf.nn.softmax(output, self.axis)
        output = tf.where(rmask, tf.constant(0.0, inputs.dtype), output)
        return output


# Copied from transformers.models.deberta.modeling_tf_deberta.get_mask
def get_mask(input, dropout):
    mask = tf.cast(
        1
        - tf.compat.v1.distributions.Bernoulli(probs=1 - dropout).sample(
            sample_shape=shape_list(input)
        ),
        tf.bool,
    )
    return mask, dropout


@tf.custom_gradient
# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaXDropout
def TFDebertaV2XDropout(input, local_ctx):
    mask, dropout = get_mask(input, local_ctx)
    scale = tf.convert_to_tensor(1.0 / (1 - dropout), dtype=input.dtype)
    # scale = tf.cast(scale, input.dtype)
    input = tf.cond(
        dropout > 0,
        lambda: tf.where(mask, tf.constant(0.0, input.dtype), input) * scale,
        lambda: input,
    )

    def custom_grad(upstream_grad):
        return tf.cond(
            scale > 1,
            lambda: (
                tf.where(mask, tf.constant(0.0, upstream_grad.dtype), upstream_grad)
                * scale,
                None,
            ),
            lambda: (upstream_grad, None),
        )

    return input, custom_grad


# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaStableDropout with Deberta->DebertaV2
class TFDebertaV2StableDropout(tf.keras.layers.Layer):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob, dtype, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = tf.convert_to_tensor(drop_prob, dtype=dtype)

    def call(self, inputs: tf.Tensor, training: tf.Tensor = False):
        if training and self.drop_prob > 0:
            return TFDebertaV2XDropout(inputs, self.drop_prob)
        return inputs


# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaSelfOutput with Deberta->DebertaV2
class TFDebertaV2SelfOutput(tf.keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.hidden_size, name="dense")
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.dropout = TFDebertaV2StableDropout(
            config.hidden_dropout_prob,
            dtype=tf.float16 if config.amp else tf.float32,
            name="dropout",
        )

    def call(self, hidden_states, input_tensor, training: bool = False):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaAttention with Deberta->DebertaV2
class TFDebertaV2Attention(tf.keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)
        self.self = TFDebertaV2DisentangledSelfAttention(config, name="self")
        self.dense_output = TFDebertaV2SelfOutput(config, name="output")
        self.config = config

    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        query_states: tf.Tensor = None,
        relative_pos: tf.Tensor = None,
        rel_embeddings: tf.Tensor = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        self_outputs = self.self(
            hidden_states=input_tensor,
            attention_mask=attention_mask,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            output_attentions=output_attentions,
            training=training,
        )
        if query_states is None:
            query_states = input_tensor
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=query_states, training=training
        )

        output = (attention_output,) + self_outputs[1:]

        return output


# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaIntermediate with Deberta->DebertaV2
class TFDebertaV2Intermediate(tf.keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaOutput with Deberta->DebertaV2
class TFDebertaV2Output(tf.keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.dropout = TFDebertaV2StableDropout(
            config.hidden_dropout_prob,
            dtype=tf.float16 if config.amp else tf.float32,
            name="dropout",
        )

    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaLayer with Deberta->DebertaV2
class TFDebertaV2Layer(tf.keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)

        self.attention = TFDebertaV2Attention(config, name="attention")
        self.intermediate = TFDebertaV2Intermediate(config, name="intermediate")
        self.bert_output = TFDebertaV2Output(config, name="output")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        query_states: tf.Tensor = None,
        relative_pos: tf.Tensor = None,
        rel_embeddings: tf.Tensor = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        attention_outputs = self.attention(
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(hidden_states=attention_output)
        layer_output = self.bert_output(
            hidden_states=intermediate_output,
            input_tensor=attention_output,
            training=training,
        )
        outputs = (layer_output,) + attention_outputs[
            1:
        ]  # add attentions if we output them

        return outputs


class TFDebertaV2ConvLayer(tf.keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)

        self.kernel_size = getattr(config, "conv_kernel_size", 3)
        # groups = getattr(config, "conv_groups", 1)
        self.conv_act = get_tf_activation(getattr(config, "conv_act", "tanh"))
        self.padding = (self.kernel_size - 1) // 2
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.dropout = TFDebertaV2StableDropout(
            config.hidden_dropout_prob,
            dtype=tf.float16 if config.amp else tf.float32,
            name="dropout",
        )
        self.config = config

    def build(self, input_shape):
        with tf.name_scope("conv"):
            self.conv_kernel = self.add_weight(
                name="kernel",
                shape=[
                    self.kernel_size,
                    self.config.hidden_size,
                    self.config.hidden_size,
                ],
                initializer=get_initializer(self.config.initializer_range),
            )
            self.conv_bias = self.add_weight(
                name="bias",
                shape=[self.config.hidden_size],
                initializer=tf.zeros_initializer(),
            )
        return super().build(input_shape)

    def call(
        self,
        hidden_states: tf.Tensor,
        residual_states: tf.Tensor,
        input_mask: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        out = tf.nn.conv2d(
            tf.expand_dims(hidden_states, 1),
            tf.expand_dims(self.conv_kernel, 0),
            strides=1,
            padding=[[0, 0], [0, 0], [self.padding, self.padding], [0, 0]],
        )
        out = tf.squeeze(tf.nn.bias_add(out, self.conv_bias), 1)
        rmask = tf.cast(1 - input_mask, tf.bool)
        out = tf.where(
            tf.broadcast_to(tf.expand_dims(rmask, -1), shape_list(out)),
            tf.cast(0.0, out.dtype),
            out,
        )
        out = self.dropout(out, training=training)
        out = self.conv_act(out)

        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input)

        if input_mask is None:
            output_states = output
        else:
            if len(shape_list(input_mask)) != len(shape_list(layer_norm_input)):
                if len(shape_list(input_mask)) == 4:
                    input_mask = tf.squeeze(tf.squeeze(input_mask, axis=1), axis=1)
                input_mask = tf.cast(tf.expand_dims(input_mask, axis=2), output.dtype)

            output_states = output * input_mask

        return output_states


class TFDebertaV2Encoder(tf.keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)

        self.layer = [
            TFDebertaV2Layer(config, name=f"layer_._{i}")
            for i in range(config.num_hidden_layers)
        ]
        self.relative_attention = getattr(config, "relative_attention", False)
        self.config = config
        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings

            self.position_buckets = getattr(config, "position_buckets", -1)
            self.pos_ebd_size = self.max_relative_positions * 2

            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets * 2

        self.norm_rel_ebd = [
            x.strip()
            for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")
        ]

        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = tf.keras.layers.LayerNormalization(
                epsilon=config.layer_norm_eps, name="LayerNorm"
            )

        self.conv = (
            TFDebertaV2ConvLayer(config, name="conv")
            if getattr(config, "conv_kernel_size", 0) > 0
            else None
        )

    def build(self, input_shape):
        if self.relative_attention:
            self.rel_embeddings = self.add_weight(
                name="rel_embeddings.weight",
                shape=[self.pos_ebd_size, self.config.hidden_size],
                initializer=get_initializer(self.config.initializer_range),
            )
        return super().build(input_shape)

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings if self.relative_attention else None
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            if self.config.amp:
                rel_embeddings = tf.cast(rel_embeddings, tf.float16)
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if len(shape_list(attention_mask)) <= 2:
            extended_attention_mask = tf.expand_dims(
                tf.expand_dims(attention_mask, 1), 2
            )
            attention_mask = extended_attention_mask * tf.expand_dims(
                tf.squeeze(extended_attention_mask, -2), -1
            )
            attention_mask = tf.cast(attention_mask, tf.uint8)
        elif len(shape_list(attention_mask)) == 3:
            attention_mask = tf.expand_dims(attention_mask, 1)

        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = (
                shape_list(query_states)[-2]
                if query_states is not None
                else shape_list(hidden_states)[-2]
            )
            relative_pos = build_relative_position(
                q,
                shape_list(hidden_states)[-2],
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        return relative_pos

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        query_states: tf.Tensor = None,
        relative_pos: tf.Tensor = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        if len(shape_list(attention_mask)) <= 2:
            input_mask = attention_mask
        else:
            input_mask = tf.cast(
                tf.math.reduce_sum(attention_mask, axis=-2) > 0, dtype=tf.uint8
            )

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        next_kv = hidden_states

        rel_embeddings = self.get_rel_embedding()
        output_states = next_kv
        for i, layer_module in enumerate(self.layer):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)

            layer_outputs = layer_module(
                hidden_states=next_kv,
                attention_mask=attention_mask,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
                output_attentions=output_attentions,
                training=training,
            )
            output_states = layer_outputs[0]

            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)

            next_kv = output_states

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)

        if not return_dict:
            return tuple(
                v
                for v in [output_states, all_hidden_states, all_attentions]
                if v is not None
            )

        return TFBaseModelOutput(
            last_hidden_state=output_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


def make_log_bucket_position(relative_pos, bucket_size, max_position):
    sign = tf.math.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = tf.where(
        (relative_pos < mid) & (relative_pos > -mid), mid - 1, tf.math.abs(relative_pos)
    )
    log_pos = (
        tf.math.ceil(
            tf.cast(tf.math.log(abs_pos / mid), tf.float32)
            / tf.math.log((max_position - 1) / mid)
            * (mid - 1)
        )
        + mid
    )
    bucket_pos = tf.cast(
        tf.where(
            abs_pos <= mid,
            tf.cast(relative_pos, tf.float32),
            log_pos * tf.cast(sign, tf.float32),
        ),
        tf.int32,
    )
    return bucket_pos


def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1):
    """
    Build relative position according to the query and key

    We assume the absolute position of query \\(P_q\\) is range from (0, query_size) and the absolute position of key
    \\(P_k\\) is range from (0, key_size), The relative positions from query to key is \\(R_{q \\rightarrow k} = P_q -
    P_k\\)

    Args:
        query_size (int): the length of query
        key_size (int): the length of key
        bucket_size (int): the size of position bucket
        max_position (int): the maximum allowed absolute position

    Return:
        `tf.Tensor`: A tensor with shape [1, query_size, key_size]

    """
    q_ids = tf.range(query_size, dtype=tf.int32)
    k_ids = tf.range(key_size, dtype=tf.int32)
    rel_pos_ids = q_ids[:, None] - tf.tile(
        tf.expand_dims(k_ids, axis=0), [shape_list(q_ids)[0], 1]
    )
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = tf.expand_dims(rel_pos_ids, axis=0)
    return tf.cast(rel_pos_ids, tf.int64)


def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    shapes = [
        shape_list(query_layer)[0],
        shape_list(query_layer)[1],
        shape_list(query_layer)[2],
        shape_list(relative_pos)[-1],
    ]
    return tf.broadcast_to(c2p_pos, shapes)


def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    shapes = [
        shape_list(query_layer)[0],
        shape_list(query_layer)[1],
        shape_list(key_layer)[-2],
        shape_list(key_layer)[-2],
    ]
    return tf.broadcast_to(c2p_pos, shapes)


def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    shapes = shape_list(p2c_att)[:2] + [
        shape_list(pos_index)[-2],
        shape_list(key_layer)[-2],
    ]
    return tf.broadcast_to(pos_index, shapes)


# https://stackoverflow.com/questions/71035337/tensorflow-create-the-torch-gather-equivalent-in-tensorflow
def torch_gather(x, indices, gather_axis):
    if gather_axis < 0:
        gather_axis = 3 + gather_axis

    all_indices = tf.where(tf.fill(indices.shape, True))
    gather_locations = tf.reshape(indices, [indices.shape.num_elements()])

    gather_indices = []
    for axis in range(len(indices.shape) - 1):
        gather_indices.append(tf.cast(all_indices[:, axis], dtype=tf.int64))

    gather_indices.append(tf.cast(gather_locations, dtype=tf.int64))

    gather_indices = tf.stack(gather_indices, axis=-1)
    gathered = tf.gather_nd(x, gather_indices)
    reshaped = tf.reshape(gathered, indices.shape)
    return reshaped


def take_along_axis(x, indices):
    # Only a valid port of np.take_along_axis when the gather axis is -1

    # TPU + gathers and reshapes don't go along well -- see https://github.com/huggingface/transformers/issues/18239
    if isinstance(tf.distribute.get_strategy(), tf.distribute.TPUStrategy):
        # [B, S, P] -> [B, S, P, D]
        one_hot_indices = tf.one_hot(indices, depth=x.shape[-1], dtype=x.dtype)

        # if we ignore the first two dims, this is equivalent to multiplying a matrix (one hot) by a vector (x)
        # grossly abusing notation: [B, S, P, D] . [B, S, D] = [B, S, P]
        gathered = tf.einsum("ijkl,ijl->ijk", one_hot_indices, x)

    # GPUs, on the other hand, prefer gathers instead of large one-hot+matmuls
    else:
        gathered = tf.gather(x, indices, batch_dims=2)

    return gathered


class TFDebertaV2DisentangledSelfAttention(tf.keras.layers.Layer):
    """
    Disentangled self-attention module

    Parameters:
        config (`DebertaV2Config`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaV2Config`]

    """

    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        _attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_head_size = getattr(
            config, "attention_head_size", _attention_head_size
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # log("attention_head_size", self.attention_head_size)
        # log("all_head_size", self.all_head_size)
        # log("num_attention_heads", self.num_attention_heads)
        self.query_proj = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query_proj",
            use_bias=True,
        )
        self.key_proj = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key_proj",
            use_bias=True,
        )
        self.value_proj = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value_proj",
            use_bias=True,
        )

        self.share_att_key = getattr(config, "share_att_key", False)
        self.pos_att_type = (
            config.pos_att_type if config.pos_att_type is not None else []
        )
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            self.pos_dropout = TFDebertaV2StableDropout(
                config.hidden_dropout_prob,
                dtype=tf.float16 if config.amp else tf.float32,
                name="pos_dropout",
            )

            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = tf.keras.layers.Dense(
                        self.all_head_size,
                        kernel_initializer=get_initializer(config.initializer_range),
                        name="pos_proj",
                        use_bias=True,
                    )
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = tf.keras.layers.Dense(
                        self.all_head_size,
                        kernel_initializer=get_initializer(config.initializer_range),
                        name="pos_q_proj",
                    )
        self.softmax = TFDebertaV2XSoftmax(axis=-1)
        self.dropout = TFDebertaV2StableDropout(
            config.attention_probs_dropout_prob,
            dtype=tf.float16 if config.amp else tf.float32,
            name="dropout",
        )
        self.amp = config.amp

    def transpose_for_scores(
        self, tensor: tf.Tensor, attention_heads: int
    ) -> tf.Tensor:
        tensor_shape = shape_list(tensor)
        # In graph mode mode, we can't reshape with -1 as the final dimension if the first dimension (batch size) is None
        shape = tensor_shape[:-1] + [
            attention_heads,
            tensor_shape[-1] // attention_heads,
        ]
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=shape)
        tensor = tf.transpose(tensor, perm=[0, 2, 1, 3])
        x_shape = shape_list(tensor)
        tensor = tf.reshape(tensor, shape=[-1, x_shape[-2], x_shape[-1]])
        return tensor

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        query_states: tf.Tensor = None,
        relative_pos: tf.Tensor = None,
        rel_embeddings: tf.Tensor = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        """
        Call the module

        Args:
            hidden_states (`tf.Tensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                *Attention(Q,K,V)*

            attention_mask (`tf.Tensor`):
                An attention mask matrix of shape [*B*, *N*, *N*] where *B* is the batch size, *N* is the maximum
                sequence length in which element [i,j] = *1* means the *i* th token in the input can attend to the *j*
                th token.

            return_att (`bool`, optional):
                Whether return the attention matrix.

            query_states (`tf.Tensor`, optional):
                The *Q* state in *Attention(Q,K,V)*.

            relative_pos (`tf.Tensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [*B*, *N*, *N*] with
                values ranging in [*-max_relative_positions*, *max_relative_positions*].

            rel_embeddings (`tf.Tensor`):
                The embedding of relative distances. It's a tensor of shape [\\(2 \\times
                \\text{max_relative_positions}\\), *hidden_size*].


        """
        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(
            self.query_proj(query_states), self.num_attention_heads
        )
        key_layer = self.transpose_for_scores(
            self.key_proj(hidden_states), self.num_attention_heads
        )
        value_layer = self.transpose_for_scores(
            self.value_proj(hidden_states), self.num_attention_heads
        )

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = tf.math.sqrt(
            tf.cast(
                shape_list(query_layer)[-1] * scale_factor,
                tf.float16 if self.amp else tf.float32,
            )
        )
        attention_scores = (
            tf.matmul(query_layer, tf.transpose(key_layer, [0, 2, 1])) / scale
        )
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_att_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )

        if rel_att is not None:
            attention_scores = attention_scores + rel_att
        attention_scores = tf.reshape(
            attention_scores,
            (
                -1,
                self.num_attention_heads,
                shape_list(attention_scores)[-2],
                shape_list(attention_scores)[-1],
            ),
        )

        # bsz x height x length x dimension
        attention_probs = self.softmax(attention_scores, attention_mask)
        attention_probs = self.dropout(attention_probs, training=training)
        context_layer = tf.matmul(
            tf.reshape(
                attention_probs,
                [-1, shape_list(attention_probs)[-2], shape_list(attention_probs)[-1]],
            ),
            value_layer,
        )
        context_layer = tf.transpose(
            tf.reshape(
                context_layer,
                [
                    -1,
                    self.num_attention_heads,
                    shape_list(context_layer)[-2],
                    shape_list(context_layer)[-1],
                ],
            ),
            [0, 2, 1, 3],
        )
        # Set the final dimension here explicitly.
        # Calling tf.reshape(context_layer, (*context_layer_shape[:-2], -1)) raises an error when executing
        # the model in graph mode as context_layer is reshaped to (None, 7, None) and Dense layer in TFDebertaV2SelfOutput
        # requires final input dimension to be defined
        context_layer_shape = shape_list(context_layer)
        new_context_layer_shape = context_layer_shape[:-2] + [
            context_layer_shape[-2] * context_layer_shape[-1]
        ]
        context_layer = tf.reshape(context_layer, new_context_layer_shape)
        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs

    def disentangled_att_bias(
        self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
    ):

        if relative_pos is None:
            q = shape_list(query_layer)[-2]
            relative_pos = build_relative_position(
                q,
                shape_list(key_layer)[-2],
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        shape_list_pos = shape_list(relative_pos)
        if len(shape_list_pos) == 2:
            relative_pos = tf.expand_dims(tf.expand_dims(relative_pos, 0), 0)
        elif len(shape_list_pos) == 3:
            relative_pos = tf.expand_dims(relative_pos, 1)
        # bsz x height x query x key
        elif len(shape_list_pos) != 4:
            raise ValueError(
                f"Relative position ids must be of dim 2 or 3 or 4. {len(shape_list_pos)}"
            )

        att_span = self.pos_ebd_size
        rel_embeddings = tf.expand_dims(
            rel_embeddings[
                self.pos_ebd_size - att_span : self.pos_ebd_size + att_span, :
            ],
            0,
        )
        if self.share_att_key:
            pos_query_layer = tf.tile(
                self.transpose_for_scores(
                    self.query_proj(rel_embeddings), self.num_attention_heads
                ),
                [shape_list(query_layer)[0] // self.num_attention_heads, 1, 1],
            )
            pos_key_layer = tf.tile(
                self.transpose_for_scores(
                    self.key_proj(rel_embeddings), self.num_attention_heads
                ),
                [shape_list(query_layer)[0] // self.num_attention_heads, 1, 1],
            )
        else:
            if "c2p" in self.pos_att_type:
                pos_key_layer = tf.tile(
                    self.transpose_for_scores(
                        self.pos_key_proj(rel_embeddings), self.num_attention_heads
                    ),
                    [shape_list(query_layer)[0] // self.num_attention_heads, 1, 1],
                )  # .split(self.all_head_size, dim=-1)
            if "p2c" in self.pos_att_type:
                pos_query_layer = tf.tile(
                    self.transpose_for_scores(
                        self.pos_query_proj(rel_embeddings), self.num_attention_heads
                    ),
                    [shape_list(query_layer)[0] // self.num_attention_heads, 1, 1],
                )  # .split(self.all_head_size, dim=-1)

        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            scale = tf.math.sqrt(
                tf.cast(
                    shape_list(pos_key_layer)[-1] * scale_factor,
                    tf.float16 if self.amp else tf.float32,
                )
            )
            c2p_att = tf.matmul(query_layer, tf.transpose(pos_key_layer, [0, 2, 1]))
            c2p_pos = tf.clip_by_value(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = take_along_axis(
                c2p_att,
                tf.broadcast_to(
                    tf.squeeze(c2p_pos, 0),
                    [
                        query_layer.shape[0],
                        query_layer.shape[1],
                        relative_pos.shape[-1],
                    ],
                ),
            )
            score += c2p_att / scale

        # position->content
        if "p2c" in self.pos_att_type:
            scale = tf.math.sqrt(
                tf.cast(
                    shape_list(pos_query_layer)[-1] * scale_factor,
                    tf.float16 if self.amp else tf.float32,
                )
            )
            if shape_list(key_layer)[-2] != shape_list(query_layer)[-2]:
                r_pos = build_relative_position(
                    shape_list(key_layer)[-2],
                    shape_list(key_layer)[-2],
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                )
                r_pos = tf.expand_dims(r_pos, 0)
            else:
                r_pos = relative_pos

            p2c_pos = tf.clip_by_value(-r_pos + att_span, 0, att_span * 2 - 1)
            # print(p2c_pos)

            p2c_att = tf.matmul(key_layer, tf.transpose(pos_query_layer, [0, 2, 1]))
            p2c_att = tf.transpose(
                take_along_axis(
                    p2c_att,
                    tf.broadcast_to(
                        tf.squeeze(p2c_pos, 0),
                        [
                            query_layer.shape[0],
                            key_layer.shape[-2],
                            key_layer.shape[-2],
                        ],
                    ),
                ),
                [0, 2, 1],
            )
            score += p2c_att / scale

        return score


class TFDebertaV3Embeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(
        self,
        config,
        shared_embeddings=None,
        input_embeddings=None,
        disentangled_gradients=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = config.vocab_size
        self.type_vocab_size = config.type_vocab_size
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.position_biased_input = getattr(config, "position_biased_input", True)
        self.initializer_range = config.initializer_range
        if self.embedding_size != config.hidden_size:
            self.embed_proj = tf.keras.layers.Dense(
                config.hidden_size,
                kernel_initializer=get_initializer(config.initializer_range),
                use_bias=False,
            )
            log("Creating projection layer")
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.dropout = TFDebertaV2StableDropout(
            config.hidden_dropout_prob,
            dtype=tf.float16 if config.amp else tf.float32,
            name="dropout",
        )
        self.disentangled_gradients = disentangled_gradients
        self.amp = config.amp

        # log("shared_embeddings: {}".format(shared_embeddings))
        log(
            "input_embeddings is {} None".format(
                "" if input_embeddings is None else "not"
            )
        )

        if shared_embeddings and input_embeddings is not None:
            log("input_embeddings.weight: ", input_embeddings.weight)
            log(
                "input_embeddings.token_type_embeddings: ",
                input_embeddings.token_type_embeddings,
            )
            log(
                "input_embeddings.position_embeddings: ",
                input_embeddings.position_embeddings,
            )
            with tf.name_scope("word_embeddings"):
                self.weight = input_embeddings.weight

            with tf.name_scope("token_type_embeddings"):
                self.token_type_embeddings = input_embeddings.token_type_embeddings

            with tf.name_scope("position_embeddings"):
                self.position_embeddings = input_embeddings.position_embeddings
            if self.disentangled_gradients:
                log("Using shared embeddings with disentangled_gradients")
                with tf.name_scope("delta_embeddings"):
                    self.delta_embeds = self.add_weight(
                        name="weight",
                        shape=[self.vocab_size, self.embedding_size],
                        initializer=tf.zeros_initializer(),
                    )
            else:
                log("Using shared embeddings without disentangled_gradients")

        else:
            log("Not sharing embeddings")
            with tf.name_scope("word_embeddings"):
                self.weight = self.add_weight(
                    name="weight",
                    shape=[self.vocab_size, self.embedding_size],
                    initializer=get_initializer(self.initializer_range),
                )

            with tf.name_scope("token_type_embeddings"):
                if self.type_vocab_size > 0:
                    self.token_type_embeddings = self.add_weight(
                        name="embeddings",
                        shape=[self.type_vocab_size, self.embedding_size],
                        initializer=get_initializer(self.initializer_range),
                    )
                else:
                    self.token_type_embeddings = None

            with tf.name_scope("position_embeddings"):
                if self.position_biased_input:
                    self.position_embeddings = self.add_weight(
                        name="embeddings",
                        shape=[self.max_position_embeddings, self.embedding_size],
                        initializer=get_initializer(self.initializer_range),
                    )
                else:
                    self.position_embeddings = None

            with tf.name_scope("delta_embeddings"):
                if self.disentangled_gradients:
                    self.delta_embeds = self.add_weight(
                        name="weight",
                        shape=[self.vocab_size, self.embedding_size],
                        initializer=tf.zeros_initializer(),
                    )
                else:
                    self.delta_embeds = None

    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        mask: tf.Tensor = None,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        assert not (input_ids is None and inputs_embeds is None)

        if input_ids is not None:
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(start=0, limit=input_shape[-1]), axis=0
            )

        if self.disentangled_gradients:
            final_embeddings = tf.stop_gradient(inputs_embeds) + tf.gather(
                params=self.delta_embeds, indices=input_ids
            )
        else:
            final_embeddings = inputs_embeds

        if self.position_biased_input:
            position_embeds = tf.gather(
                params=self.position_embeddings, indices=position_ids
            )
            final_embeddings += position_embeds

        if self.type_vocab_size > 0:
            token_type_embeds = tf.gather(
                params=self.token_type_embeddings, indices=token_type_ids
            )
            final_embeddings += token_type_embeds

        if self.embedding_size != self.hidden_size:
            final_embeddings = self.embed_proj(final_embeddings)

        final_embeddings = self.LayerNorm(final_embeddings)

        if mask is not None:
            if len(shape_list(mask)) != len(shape_list(final_embeddings)):
                if len(shape_list(mask)) == 4:
                    mask = tf.squeeze(tf.squeeze(mask, axis=1), axis=1)
                mask = tf.cast(tf.expand_dims(mask, axis=2), final_embeddings.dtype)

            final_embeddings = final_embeddings * mask

        final_embeddings = self.dropout(final_embeddings, training=training)

        return final_embeddings

    def update_embeddings(self):
        # hack for scope renameing, and merging the delta embeddings
        # the scope is needed since the embedding weight were under the wrond scope
        # since they were not created in build but in __init__
        with tf.name_scope("tf_deberta_v2_model/deberta/embeddings/word_embeddings"):
            if self.disentangled_gradients:
                log(" ** Discriminator was trained using disentangled gradients")
                self.weight = tf.Variable(
                    self.weight + self.delta_embeds,
                    name="weight",
                )
                del self.delta_embeds
            else:
                log(" ** Discriminator was trained without disentangled gradients")
                self.weight = tf.Variable(self.weight, name="weight")

        if self.type_vocab_size > 0:
            with tf.name_scope(
                "tf_deberta_v2_model/deberta/embeddings/token_type_embeddings"
            ):
                self.token_type_embeddings = tf.Variable(
                    self.token_type_embeddings, name="embeddings"
                )

        if self.position_biased_input:
            with tf.name_scope(
                "tf_deberta_v2_model/deberta/embeddings/position_embeddings"
            ):
                self.position_embeddings = tf.Variable(
                    self.position_embeddings, name="embeddings"
                )


# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaPredictionHeadTransform with Deberta->DebertaV2
class TFDebertaV2PredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.embedding_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaLMPredictionHead with Deberta->DebertaV2
class TFDebertaV2LMPredictionHead(tf.keras.layers.Layer):
    def __init__(
        self, config: DebertaV2Config, input_embeddings: tf.keras.layers.Layer, **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size

        self.transform = TFDebertaV2PredictionHeadTransform(config, name="transform")

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.input_embeddings = input_embeddings

    def build(self, input_shape: tf.TensorShape):
        self.bias = self.add_weight(
            shape=(self.vocab_size,), initializer="zeros", trainable=True, name="bias"
        )

        super().build(input_shape)

    def get_output_embeddings(self) -> tf.keras.layers.Layer:
        return self.input_embeddings

    def set_output_embeddings(self, value: tf.Variable):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"bias": self.bias}

    def set_bias(self, value: tf.Variable):
        self.bias = value["bias"]
        self.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.transform(hidden_states=hidden_states)
        seq_length = shape_list(hidden_states)[1]
        hidden_states = tf.reshape(
            tensor=hidden_states, shape=[-1, self.embedding_size]
        )
        hidden_states = tf.matmul(
            a=hidden_states, b=self.input_embeddings.weight, transpose_b=True
        )
        hidden_states = tf.reshape(
            tensor=hidden_states, shape=[-1, seq_length, self.vocab_size]
        )
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states


# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaOnlyMLMHead with Deberta->DebertaV2
class TFDebertaV2OnlyMLMHead(tf.keras.layers.Layer):
    def __init__(
        self, config: DebertaV2Config, input_embeddings: tf.keras.layers.Layer, **kwargs
    ):
        super().__init__(**kwargs)
        self.predictions = TFDebertaV2LMPredictionHead(
            config, input_embeddings, name="predictions"
        )

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        prediction_scores = self.predictions(hidden_states=sequence_output)

        return prediction_scores


# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaMainLayer with Deberta->DebertaV2
class TFDebertaV2MainLayer(tf.keras.layers.Layer):
    config_class = DebertaV2Config

    def __init__(
        self,
        config: DebertaV2Config,
        shared_embeddings=False,
        input_embeddings=None,
        disentangled_gradients=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.config = config
        kwargs.pop("name", None)

        self.embeddings = TFDebertaV3Embeddings(
            config=config,
            name="embeddings",
            shared_embeddings=shared_embeddings,
            input_embeddings=input_embeddings,
            disentangled_gradients=disentangled_gradients,
        )

        self.encoder = TFDebertaV2Encoder(config, name="encoder")

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def update_embeddings(self):
        self.embeddings.update_embeddings()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            mask=attention_mask,
            training=training,
        )

        # if hasattr(self, "embeddings_project"):
        #     embedding_output = self.embeddings_project(
        #         embedding_output, training=training
        #     )

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return TFBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaPreTrainedModel with Deberta->DebertaV2
class TFDebertaV2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DebertaV2Config
    base_model_prefix = "deberta"


DEBERTA_START_DOCSTRING = r"""
    The DeBERTa model was proposed in [DeBERTa: Decoding-enhanced BERT with Disentangled
    Attention](https://arxiv.org/abs/2006.03654) by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen. It's build
    on top of BERT/RoBERTa with two improvements, i.e. disentangled attention and enhanced mask decoder. With those two
    improvements, it out perform BERT/RoBERTa on a majority of tasks with 80GB pretraining data.

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TF 2.0 models accepts two formats as inputs:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional arguments.

    This second option is useful when using [`tf.keras.Model.fit`] method which currently requires having all the
    tensors in the first argument of the model call function: `model(inputs)`.

    If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
    first positional argument :

    - a single Tensor with `input_ids` only and nothing else: `model(inputs_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    </Tip>

    Parameters:
        config ([`DebertaV2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DEBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`DebertaV2Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        inputs_embeds (`np.ndarray` or `tf.Tensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~transformers.file_utils.ModelOutput``] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare DeBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    DEBERTA_START_DOCSTRING,
)
# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaModel with Deberta->DebertaV2
class TFDebertaV2Model(TFDebertaV2PreTrainedModel):
    def __init__(
        self,
        config: DebertaV2Config,
        shared_embeddings=False,
        input_embeddings=None,
        disentangled_gradients=False,
        *inputs,
        **kwargs,
    ):
        super().__init__(
            config,
            *inputs,
            **kwargs,
        )

        self.deberta = TFDebertaV2MainLayer(
            config,
            shared_embeddings=shared_embeddings,
            input_embeddings=input_embeddings,
            disentangled_gradients=disentangled_gradients,
            **kwargs,
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def serving_output(self, output: TFBaseModelOutput) -> TFBaseModelOutput:
        hs = (
            tf.convert_to_tensor(output.hidden_states)
            if self.config.output_hidden_states
            else None
        )
        attns = (
            tf.convert_to_tensor(output.attentions)
            if self.config.output_attentions
            else None
        )

        return TFBaseModelOutput(
            last_hidden_state=output.last_hidden_state,
            hidden_states=hs,
            attentions=attns,
        )

    def update_embeddings(self):
        self.deberta.embeddings.update_embeddings()


@add_start_docstrings(
    """DeBERTa Model with a `language modeling` head on top.""", DEBERTA_START_DOCSTRING
)
# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaForMaskedLM with Deberta->DebertaV2
class TFDebertaV2ForMaskedLM(TFDebertaV2PreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(
        self,
        config: DebertaV2Config,
        shared_embeddings=False,
        input_embeddings=None,
        *inputs,
        **kwargs,
    ):
        super().__init__(config, *inputs, **kwargs)

        if config.is_decoder:
            logger.warning(
                "If you want to use `TFDebertaV2ForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.deberta = TFDebertaV2MainLayer(
            config,
            shared_embeddings=shared_embeddings,
            input_embeddings=input_embeddings,
            name="deberta",
            **kwargs,
        )
        self.mlm = TFDebertaV2OnlyMLMHead(
            config, input_embeddings=self.deberta.embeddings, name="cls"
        )

    def get_lm_head(self) -> tf.keras.layers.Layer:
        return self.mlm.predictions

    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = outputs[0]
        prediction_scores = self.mlm(sequence_output=sequence_output, training=training)
        loss = (
            None
            if labels is None
            else self.hf_compute_loss(labels=labels, logits=prediction_scores)
        )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(self, output: TFMaskedLMOutput) -> TFMaskedLMOutput:
        hs = (
            tf.convert_to_tensor(output.hidden_states)
            if self.config.output_hidden_states
            else None
        )
        attns = (
            tf.convert_to_tensor(output.attentions)
            if self.config.output_attentions
            else None
        )

        return TFMaskedLMOutput(
            logits=output.logits, hidden_states=hs, attentions=attns
        )

    def update_embeddings(self):
        self.deberta.embeddings.update_embeddings()


@add_start_docstrings(
    """
    DeBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaForSequenceClassification with Deberta->DebertaV2
class TFDebertaV2ForSequenceClassification(
    TFDebertaV2PreTrainedModel, TFSequenceClassificationLoss
):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        self.deberta = TFDebertaV2MainLayer(config, name="deberta")
        self.pooler = TFDebertaV2ContextPooler(config, name="pooler")

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = TFDebertaV2StableDropout(
            drop_out, dtype=tf.float16 if config.amp else tf.float32, name="cls_dropout"
        )
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output, training=training)
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        loss = (
            None
            if labels is None
            else self.hf_compute_loss(labels=labels, logits=logits)
        )

        if not return_dict:
            output = (logits,) + outputs[1:]

            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(
        self, output: TFSequenceClassifierOutput
    ) -> TFSequenceClassifierOutput:
        hs = (
            tf.convert_to_tensor(output.hidden_states)
            if self.config.output_hidden_states
            else None
        )
        attns = (
            tf.convert_to_tensor(output.attentions)
            if self.config.output_attentions
            else None
        )

        return TFSequenceClassifierOutput(
            logits=output.logits, hidden_states=hs, attentions=attns
        )


@add_start_docstrings(
    """
    DeBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaForTokenClassification with Deberta->DebertaV2
class TFDebertaV2ForTokenClassification(
    TFDebertaV2PreTrainedModel, TFTokenClassificationLoss
):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        self.deberta = TFDebertaV2MainLayer(config, name="deberta")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output, training=training)
        logits = self.classifier(inputs=sequence_output)
        loss = (
            None
            if labels is None
            else self.hf_compute_loss(labels=labels, logits=logits)
        )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(
        self, output: TFTokenClassifierOutput
    ) -> TFTokenClassifierOutput:
        hs = (
            tf.convert_to_tensor(output.hidden_states)
            if self.config.output_hidden_states
            else None
        )
        attns = (
            tf.convert_to_tensor(output.attentions)
            if self.config.output_attentions
            else None
        )

        return TFTokenClassifierOutput(
            logits=output.logits, hidden_states=hs, attentions=attns
        )


@add_start_docstrings(
    """
    DeBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    DEBERTA_START_DOCSTRING,
)
# Copied from transformers.models.deberta.modeling_tf_deberta.TFDebertaForQuestionAnswering with Deberta->DebertaV2
class TFDebertaV2ForQuestionAnswering(
    TFDebertaV2PreTrainedModel, TFQuestionAnsweringLoss
):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        self.deberta = TFDebertaV2MainLayer(config, name="deberta")
        self.qa_outputs = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="qa_outputs",
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: Optional[Union[np.ndarray, tf.Tensor]] = None,
        end_positions: Optional[Union[np.ndarray, tf.Tensor]] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        start_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = outputs[0]
        logits = self.qa_outputs(inputs=sequence_output)
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        loss = None

        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(
                labels=labels, logits=(start_logits, end_logits)
            )

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(
        self, output: TFQuestionAnsweringModelOutput
    ) -> TFQuestionAnsweringModelOutput:
        hs = (
            tf.convert_to_tensor(output.hidden_states)
            if self.config.output_hidden_states
            else None
        )
        attns = (
            tf.convert_to_tensor(output.attentions)
            if self.config.output_attentions
            else None
        )

        return TFQuestionAnsweringModelOutput(
            start_logits=output.start_logits,
            end_logits=output.end_logits,
            hidden_states=hs,
            attentions=attns,
        )


class TFDebertaV3DiscriminatorPredictions(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(config.hidden_size, name="dense")
        self.dense_prediction = tf.keras.layers.Dense(1, name="dense_prediction")
        self.config = config

    def call(self, discriminator_hidden_states, training=False):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_tf_activation(self.config.hidden_act)(hidden_states)
        logits = tf.squeeze(self.dense_prediction(hidden_states), -1)

        return logits


@dataclass
class TFDebertaV3ForRTDOutput(ModelOutput):
    """
    Output type of [`TFDebertaV2ForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `tf.Tensor` of shape `(1,)`):
            Total loss of the DebertaV3 objective.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


@add_start_docstrings(
    """
    Deberta-v2 model with a binary classification head on top as used during pretraining for identifying generated tokens.

    Even though both the discriminator and generator may be loaded into this model, the discriminator is the only model
    of the two to have the correct classification head to be used for this model.
    """,
    DEBERTA_START_DOCSTRING,
)
class TFDebertaV3ForRTD(TFDebertaV2PreTrainedModel):
    def __init__(
        self,
        config,
        shared_embeddings=False,
        input_embeddings=None,
        disentangled_gradients=False,
        **kwargs,
    ):
        super().__init__(
            config,
            **kwargs,
        )

        self.debertav2 = TFDebertaV2Model(
            config,
            name="deberta",
            shared_embeddings=shared_embeddings,
            input_embeddings=input_embeddings,
            disentangled_gradients=disentangled_gradients,
            **kwargs,
        )
        self.discriminator_predictions = TFDebertaV3DiscriminatorPredictions(
            config, name="discriminator_predictions"
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TFDebertaV3ForRTDOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[TFDebertaV3ForRTDOutput, Tuple[tf.Tensor]]:

        outputs = self.debertav2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        discriminator_sequence_output = outputs.last_hidden_state

        logits = self.discriminator_predictions(discriminator_sequence_output)

        if not return_dict:
            return (logits,) + outputs[1:]

        return TFDebertaV3ForRTDOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(self, output):
        hs = (
            tf.convert_to_tensor(output.hidden_states)
            if self.config.output_hidden_states
            else None
        )
        attns = (
            tf.convert_to_tensor(output.attentions)
            if self.config.output_attentions
            else None
        )

        return TFDebertaV3ForRTDOutput(
            logits=output.logits, hidden_states=hs, attentions=attns
        )


def get_generator_config(config, bert_config):
    """Get model config for the generator network."""
    gen_config = DebertaV2Config.from_dict(bert_config.to_dict())
    gen_config.hidden_size = int(
        round(bert_config.hidden_size * config.generator_hidden_size)
    )
    # To keep hidden size divisble by 64 - attention head size
    if gen_config.hidden_size % 64 != 0:
        gen_config.hidden_size += 64 - (gen_config.hidden_size % 64)
    gen_config.num_hidden_layers = int(
        round(bert_config.num_hidden_layers * config.generator_layers)
    )
    gen_config.intermediate_size = 4 * gen_config.hidden_size
    gen_config.num_attention_heads = max(1, gen_config.hidden_size // 64)
    return gen_config


class PretrainingModel(tf.keras.Model):
    """Transformer pre-training using the replaced-token-detection task.
    Sourced from https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow2/LanguageModeling/ELECTRA/modeling.py
    """

    def __init__(self, config: DebertaV3PretrainingConfig, **kwargs):
        super().__init__(**kwargs)
        # Set up model config
        self._config = config
        self.disc_config = DebertaV2Config(
            model_name=config.model_name,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            embedding_size=config.embedding_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=4 * config.hidden_size,
            hidden_act=config.hidden_act,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            relative_attention=config.relative_attention,
            position_buckets=config.position_buckets,
            position_biased_input=config.position_biased_input,
        )
        self.disc_config.update({"amp": config.amp})
        # self.disc_config.update({"output_hidden_states": True})

        if config.electra_objective:

            # Set up generator
            gen_config = get_generator_config(config, self.disc_config)
            gen_config.update({"amp": config.amp})
            heading("Generator Config")
            log_config(gen_config)
            self.generator = TFDebertaV2ForMaskedLM(gen_config)

            # Set up discriminator
            if config.shared_embeddings:
                log("Shared embeddings with Electra Objective")
                self.discriminator = TFDebertaV3ForRTD(
                    self.disc_config,
                    shared_embeddings=True,
                    input_embeddings=self.generator.deberta.get_input_embeddings(),
                    disentangled_gradients=config.disentangled_gradients,
                )
            else:
                self.discriminator = TFDebertaV3ForRTD(self.disc_config)
        else:
            self.generator = TFDebertaV2ForMaskedLM(self.disc_config)

    def call(self, features, is_training):
        config = self._config

        # print("features", features)
        # Mask the input
        unmasked_inputs = pretrain_utils.features_to_inputs(features)
        masked_inputs = pretrain_utils.mask(config, unmasked_inputs, config.mask_prob)

        # Generator
        if config.uniform_generator:
            mlm_output = self._get_masked_lm_output(
                masked_inputs, None, is_training=is_training
            )
        else:
            mlm_output = self._get_masked_lm_output(
                masked_inputs, self.generator, is_training=is_training
            )
        fake_data = self._get_fake_data(masked_inputs, mlm_output.logits)
        total_loss = config.gen_weight * mlm_output.loss

        # Discriminator
        disc_output = None
        if config.electra_objective:
            disc_output = self._get_discriminator_output(
                fake_data.inputs,
                self.discriminator,
                fake_data.is_fake_tokens,
                is_training=is_training,
            )
            total_loss += config.disc_weight * disc_output.loss

        # Evaluation inputs
        eval_fn_inputs = {
            "input_ids": masked_inputs.input_ids,
            "masked_lm_preds": mlm_output.preds,
            "mlm_loss": mlm_output.per_example_loss,
            "masked_lm_ids": masked_inputs.masked_lm_ids,
            "masked_lm_weights": masked_inputs.masked_lm_weights,
            "input_mask": masked_inputs.input_mask,
        }
        if config.electra_objective:
            eval_fn_inputs.update(
                {
                    "disc_loss": disc_output.per_example_loss,
                    "disc_labels": disc_output.labels,
                    "disc_probs": disc_output.probs,
                    "disc_preds": disc_output.preds,
                    "sampled_tokids": tf.argmax(
                        fake_data.sampled_tokens, -1, output_type=tf.int32
                    ),
                }
            )

        return total_loss, eval_fn_inputs

    def _get_masked_lm_output(self, inputs, generator, is_training=False):
        """Masked language modeling softmax layer."""
        masked_lm_weights = inputs.masked_lm_weights

        if self._config.uniform_generator:
            logits = tf.zeros(self.disc_config.vocab_size)
            logits_tiled = tf.zeros(
                pretrain_utils.get_shape_list(inputs.masked_lm_ids)
                + [self.disc_config.vocab_size]
            )
            logits_tiled += tf.reshape(logits, [1, 1, self.disc_config.vocab_size])
            logits = logits_tiled
        else:
            outputs = generator(
                input_ids=inputs.input_ids,
                attention_mask=inputs.input_mask,
                token_type_ids=inputs.segment_ids,
                training=is_training,
            )
            logits = outputs[0]
            logits = pretrain_utils.gather_positions(logits, inputs.masked_lm_positions)

        oh_labels = tf.one_hot(
            inputs.masked_lm_ids, depth=self.disc_config.vocab_size, dtype=tf.float32
        )

        probs = tf.cast(tf.nn.softmax(logits), tf.float32)
        log_probs = tf.cast(tf.nn.log_softmax(logits), tf.float32)
        label_log_probs = -tf.reduce_sum(log_probs * oh_labels, axis=-1)

        numerator = tf.reduce_sum(masked_lm_weights * label_log_probs)
        denominator = tf.reduce_sum(masked_lm_weights) + 1e-6
        loss = numerator / denominator
        preds = tf.argmax(log_probs, axis=-1, output_type=tf.int32)

        MLMOutput = collections.namedtuple(
            "MLMOutput", ["logits", "probs", "loss", "per_example_loss", "preds"]
        )
        return MLMOutput(
            logits=logits,
            probs=probs,
            per_example_loss=label_log_probs,
            loss=loss,
            preds=preds,
        )

    def _get_discriminator_output(
        self, inputs, discriminator, labels, is_training=False
    ):
        """Discriminator binary classifier."""

        outputs = discriminator(
            input_ids=inputs.input_ids,
            attention_mask=inputs.input_mask,
            token_type_ids=inputs.segment_ids,
            training=is_training,
        )
        logits = outputs.logits
        weights = tf.cast(inputs.input_mask, tf.float32)
        labelsf = tf.cast(labels, tf.float32)
        logits = tf.cast(logits, tf.float32)
        losses = (
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labelsf)
            * weights
        )
        per_example_loss = tf.reduce_sum(losses, axis=-1) / (
            1e-6 + tf.reduce_sum(weights, axis=-1)
        )
        loss = tf.reduce_sum(losses) / (1e-6 + tf.reduce_sum(weights))
        probs = tf.nn.sigmoid(logits)
        preds = tf.cast(tf.round((tf.sign(logits) + 1) / 2), tf.int32)
        DiscOutput = collections.namedtuple(
            "DiscOutput", ["loss", "per_example_loss", "probs", "preds", "labels"]
        )
        return DiscOutput(
            loss=loss,
            per_example_loss=per_example_loss,
            probs=probs,
            preds=preds,
            labels=labels,
        )

    def _get_fake_data(self, inputs, mlm_logits):
        """Sample from the generator to create corrupted input."""
        inputs = pretrain_utils.unmask(inputs)
        disallow = (
            tf.one_hot(
                inputs.masked_lm_ids,
                depth=self.disc_config.vocab_size,
                dtype=tf.float32,
            )
            if self._config.disallow_correct
            else None
        )
        sampled_tokens = tf.stop_gradient(
            pretrain_utils.sample_from_softmax(
                mlm_logits / self._config.temperature, disallow=disallow
            )
        )
        sampled_tokids = tf.argmax(sampled_tokens, -1, output_type=tf.int32)
        updated_input_ids, masked = pretrain_utils.scatter_update(
            inputs.input_ids, sampled_tokids, inputs.masked_lm_positions
        )
        labels = masked * (
            1 - tf.cast(tf.equal(updated_input_ids, inputs.input_ids), tf.int32)
        )
        updated_inputs = pretrain_utils.get_updated_inputs(
            inputs, input_ids=updated_input_ids
        )
        FakedData = collections.namedtuple(
            "FakedData", ["inputs", "is_fake_tokens", "sampled_tokens"]
        )
        return FakedData(
            inputs=updated_inputs, is_fake_tokens=labels, sampled_tokens=sampled_tokens
        )
