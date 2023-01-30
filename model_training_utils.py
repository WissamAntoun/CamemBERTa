# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
"""A light weight utilities to train NLP models."""

from __future__ import absolute_import, division, print_function

import datetime
import json
import os
import time

import tensorflow as tf
from horovod.tensorflow.compression import Compression

import utils
from configuration_deberta_v2 import DebertaV3PretrainingConfig
from modeling_tf_deberta_v2 import PretrainingModel as DebertaPretrainingModel
from modeling_tf_roberta import PretrainingModel as RobertaPretrainingModel
from official_utils.misc import distribution_utils
from optimization import GradientAccumulatorv2
from utils import log


_SUMMARY_TXT = "training_summary.txt"
_MIN_SUMMARY_STEPS = 10

# tf.get_logger().setLevel("DEBUG")
# logging = tf.get_logger()
# logging.propagate = False


def _save_checkpoint(checkpoint, model_dir, checkpoint_prefix):
    """Saves model to with provided checkpoint prefix."""

    checkpoint_path = os.path.join(model_dir, checkpoint_prefix)
    saved_path = checkpoint.save(checkpoint_path)
    log("Saving model as TF checkpoint: {}".format(saved_path))
    return


def _get_input_iterator(input_fn, strategy):
    """Returns distributed dataset iterator."""
    # When training with TPU pods, datasets needs to be cloned across
    # workers. Since Dataset instance cannot be cloned in eager mode, we instead
    # pass callable that returns a dataset.
    if not callable(input_fn):
        raise ValueError("`input_fn` should be a closure that returns a dataset.")
    if strategy is None:
        input_data = input_fn()
        iterator = iter(input_data)
    else:
        iterator = iter(strategy.distribute_datasets_from_function(input_fn))
    return iterator


def _float_metric_value(metric):
    """Gets the value of a float-value keras metric."""
    return metric.result().numpy().astype(float)


def steps_to_run(current_step, eval_every_n_steps, steps_per_loop):
    """Calculates steps to run on device."""
    if steps_per_loop <= 0:
        raise ValueError("steps_per_loop should be positive integer.")
    if steps_per_loop == 1:
        return steps_per_loop
    remainder_in_epoch = current_step % eval_every_n_steps
    if remainder_in_epoch != 0:
        return min(eval_every_n_steps - remainder_in_epoch, steps_per_loop)
    else:
        return steps_per_loop


def write_txt_summary(training_summary, summary_dir):
    """Writes a summary text file to record stats."""
    summary_path = os.path.join(summary_dir, _SUMMARY_TXT)
    with tf.io.gfile.GFile(summary_path, "wb") as f:
        log("Training Summary: \n{}".format(str(training_summary)))
        f.write(json.dumps(training_summary, indent=4))


def run_customized_training_loop(
    # pylint: disable=invalid-name
    _sentinel=None,
    # pylint: enable=invalid-name
    strategy: tf.distribute.Strategy = None,
    model_fn=None,
    config: DebertaV3PretrainingConfig = None,
    loss_fn=None,
    train_input_fn=None,
    eval_input_fn=None,
    eval_steps=None,
    metric_fn_dict=None,
    custom_callbacks=None,
    run_eagerly=False,
    hvd=None,
):
    """Run BERT pretrain model training using low-level API.

    Arguments:
        _sentinel: Used to prevent positional parameters. Internal, do not use.
        strategy: Distribution strategy on which to run low level training loop.
        config: a DebertaV3PretrainingConfig object
        model_fn: Function that returns a tuple (model, sub_model). Caller of this
          function should add optimizer to the `model` via calling
          `model.compile()` API or manually setting `model.optimizer` attribute.
          Second element of the returned tuple(sub_model) is an optional sub model
          to be used for initial checkpoint -- if provided.
        loss_fn: Function with signature func(labels, logits) and returns a loss
          tensor.
        train_input_fn: Function that returns a tf.data.Dataset used for training.
        eval_input_fn: Function that returns evaluation dataset. If none,
          evaluation is skipped.
        eval_steps: Number of steps to run evaluation. Required if `eval_input_fn`
          is not none.
        metric_fn_dict: A metrics dictionary with keyes as metric string names,
          and values as function that returns a Keras Metric object to record
          evaluation result using evaluation dataset or with training dataset
          after every epoch.
        custom_callbacks: A list of Keras Callbacks objects to run during
          training. More specifically, `on_batch_begin()`, `on_batch_end()`,
          methods are invoked during training.
        run_eagerly: Whether to run model training in pure eager execution. This
          should be disable for TPUStrategy.


    Returns:
        Trained model.

    Raises:
        ValueError: (1) When model returned by `model_fn` does not have optimizer
          attribute or when required parameters are set to none. (2) eval args are
          not specified correctly. (3) metric_fn_dict must be a callable if specified.
          (4) sub_model_checkpoint_name is specified, but `sub_model` returned
          by `model_fn` is None.
    """

    if _sentinel is not None:
        raise ValueError(
            "only call `run_customized_training_loop()` " "with named arguments."
        )

    required_arguments = [model_fn, loss_fn, config, train_input_fn]
    if [arg for arg in required_arguments if arg is None]:
        raise ValueError(
            "`model_fn`, `loss_fn`, `config`, " "are required " "parameters."
        )

    eval_every_n_steps = config.eval_every_n_steps
    steps_per_loop = config.log_freq

    if eval_every_n_steps > 0 and steps_per_loop > eval_every_n_steps:
        log(
            "ERROR:\n steps_per_loop: {} is specified to be greater than "
            " eval_every_n_steps: {}, we will use eval_every_n_steps as"
            " steps_per_loop.".format(steps_per_loop, eval_every_n_steps)
        )
        steps_per_loop = eval_every_n_steps
    assert tf.executing_eagerly()

    if run_eagerly:
        if steps_per_loop > 1:
            raise ValueError(
                "steps_per_loop is used for performance optimization. When you want "
                "to run eagerly, you cannot leverage graph mode loop."
            )
        if isinstance(strategy, tf.distribute.TPUStrategy):
            raise ValueError(
                "TPUStrategy should not run eagerly as it heavily replies on graph"
                " optimization for the distributed system."
            )

    if eval_input_fn and (eval_steps is None or metric_fn_dict is None):
        raise ValueError(
            "`eval_step` and `metric_fn_dict` are required when `eval_input_fn ` "
            "is not none."
        )
    # if metric_fn_dict and not callable(metric_fn_dict):
    #     raise ValueError("if `metric_fn_dict` is specified, metric_fn_dict must be a callable.")

    total_training_steps = config.num_train_steps
    num_accumulative_step = config.gradient_accumulation_steps
    # To reduce unnecessary send/receive input pipeline operation, we place input
    # pipeline ops in worker task.
    train_iterator = _get_input_iterator(train_input_fn, strategy)

    with distribution_utils.get_strategy_scope(strategy):
        # To correctly place the model weights on accelerators,
        # model and optimizer should be created in scope.
        model: DebertaPretrainingModel = None
        model, optimizer = model_fn()

        first_batch = True

        use_float16 = config.amp

        eval_metrics = metric_fn_dict
        # If evaluation is required, make a copy of metric as it will be used by
        # both train and evaluation.
        train_metrics = {
            key: metric.__class__.from_config(metric.get_config())
            for key, metric in eval_metrics.items()
        }

        train_loss_metric = train_metrics["total_loss"]

        # Create summary writers
        if not hvd or hvd.rank() == 0:
            summary_dir = os.path.join(config.checkpoints_dir, "summaries")
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            eval_summary_writer = tf.summary.create_file_writer(
                os.path.join(
                    summary_dir, "eval", "p2" if config.phase2 else "p1", current_time
                )
            )
            if steps_per_loop >= _MIN_SUMMARY_STEPS:
                # Only writes summary when the stats are collected sufficiently over
                # enough steps.
                train_summary_writer = tf.summary.create_file_writer(
                    os.path.join(
                        summary_dir,
                        "train",
                        "p2" if config.phase2 else "p1",
                        current_time,
                    )
                )
            else:
                train_summary_writer = None
        else:
            eval_summary_writer = None
            train_summary_writer = None
            eval_input_fn = None

        accum_gradients = GradientAccumulatorv2()

        def run_metric_fn(config: DebertaV3PretrainingConfig, metrics, eval_fn_inputs):
            """Computes the loss and accuracy of the model."""
            d = eval_fn_inputs
            metrics["masked_lm_accuracy"].update_state(
                y_true=tf.reshape(d["masked_lm_ids"], [-1]),
                y_pred=tf.reshape(d["masked_lm_preds"], [-1]),
                sample_weight=tf.reshape(d["masked_lm_weights"], [-1]),
            )
            metrics["masked_lm_loss"].update_state(
                values=tf.reshape(d["mlm_loss"], [-1]),
                sample_weight=tf.reshape(d["masked_lm_weights"], [-1]),
            )
            if config.electra_objective:
                metrics["sampled_masked_lm_accuracy"].update_state(
                    y_true=tf.reshape(d["masked_lm_ids"], [-1]),
                    y_pred=tf.reshape(d["sampled_tokids"], [-1]),
                    sample_weight=tf.reshape(d["masked_lm_weights"], [-1]),
                )
                if config.disc_weight > 0:
                    metrics["disc_loss"].update_state(d["disc_loss"])
                    # metrics["disc_auc"].update_state(
                    #    d["disc_labels"] * d["input_mask"],
                    #    d["disc_probs"] * tf.cast(d["input_mask"], tf.float32))
                    metrics["disc_accuracy"].update_state(
                        y_true=d["disc_labels"],
                        y_pred=d["disc_preds"],
                        sample_weight=d["input_mask"],
                    )
                    metrics["disc_precision"].update_state(
                        y_true=d["disc_labels"],
                        y_pred=d["disc_preds"],
                        sample_weight=d["disc_preds"] * d["input_mask"],
                    )
                    metrics["disc_recall"].update_state(
                        y_true=d["disc_labels"],
                        y_pred=d["disc_preds"],
                        sample_weight=d["disc_labels"] * d["input_mask"],
                    )
            return metrics

        def _replicated_step(inputs, first_batch=False):
            """Replicated training step."""

            with tf.GradientTape() as tape:
                loss, model_outputs = model(inputs, is_training=True)
                loss = loss_fn(loss)
                if use_float16:
                    scaled_loss = optimizer.get_scaled_loss(loss)

            if hvd:
                tape = hvd.DistributedGradientTape(
                    tape,
                    sparse_as_dense=True,
                    compression=Compression.fp16
                    if config.fp16_compression
                    else Compression.none,
                )

            # Collects training variables.

            if use_float16:
                scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
                grads = optimizer.get_unscaled_gradients(scaled_grads)
            else:
                grads = tape.gradient(loss, model.trainable_variables)

            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if hvd and first_batch:
                hvd.broadcast_variables(model.variables, 0)
                hvd.broadcast_variables(optimizer.variables(), 0)

            # For reporting, the metric takes the mean of losses.
            train_loss_metric.update_state(loss)
            run_metric_fn(config, train_metrics, model_outputs)

        @tf.function(jit_compile=config.xla)
        def _forward(inputs):
            with tf.GradientTape() as tape:
                loss, model_outputs = model(inputs, is_training=True)
                if use_float16:
                    scaled_loss = optimizer.get_scaled_loss(loss)

            if use_float16:
                scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
                grads = optimizer.get_unscaled_gradients(scaled_grads)
            else:
                grads = tape.gradient(loss, model.trainable_variables)

            # For reporting, the metric takes the mean of losses.
            train_loss_metric.update_state(tf.stop_gradient(loss))
            run_metric_fn(config, train_metrics, model_outputs)
            accum_gradients.add_gradients(grads)

        def _step(num_grad_accumulates):
            if hvd:
                gradients = [
                    None
                    if g is None
                    else hvd.allreduce(
                        g / tf.cast(num_grad_accumulates, g.dtype),
                        compression=Compression.fp16
                        if config.fp16_compression
                        else Compression.none,
                    )
                    for g in accum_gradients.gradients
                ]
            else:
                gradients = [
                    None if g is None else g / tf.cast(num_grad_accumulates, g.dtype)
                    for g in accum_gradients.gradients
                ]
            (gradients, _) = tf.clip_by_global_norm(gradients, clip_norm=1.0)

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            accum_gradients.reset()

        @tf.function
        def train_steps_strategy(iterator, steps, num_grad_accumulates):
            """Performs distributed training steps in a loop.

            Args:
              iterator: the distributed iterator of training datasets.
              steps: an tf.int32 integer tensor to specify number of steps to run
                inside host training loop.

            Raises:
              ValueError: Any of the arguments or tensor shapes are invalid.
            """
            if not isinstance(steps, tf.Tensor):
                raise ValueError(
                    "steps should be an Tensor. Python object may cause " "retracing."
                )

            if num_grad_accumulates != 1:
                for step_idx in tf.range(steps * num_grad_accumulates):
                    # logging.info("ACC Step %d", step_idx)
                    strategy.run(_forward, args=(next(iterator),))
                    if (step_idx + 1) % num_grad_accumulates == 0:
                        strategy.run(_step, args=(num_grad_accumulates,))
            else:
                for step_idx in tf.range(steps):
                    strategy.run(_replicated_step, args=(next(iterator),))

        @tf.function
        def train_steps(iterator, steps, num_grad_accumulates, first_batch):
            if not isinstance(steps, tf.Tensor):
                raise ValueError(
                    "steps should be an Tensor. Python object may cause " "retracing."
                )

            if num_grad_accumulates != 1:
                for step_idx in tf.range(steps * num_grad_accumulates):
                    _forward(next(iterator))
                    if (step_idx + 1) % num_grad_accumulates == 0:
                        _step(num_grad_accumulates)
                    if hvd and step_idx == 0 and first_batch:
                        hvd.broadcast_variables(model.variables, 0)
                        hvd.broadcast_variables(optimizer.variables(), 0)
            else:
                for step_idx in tf.range(steps):
                    _replicated_step(next(iterator), (first_batch and step_idx == 0))

        @tf.function
        def train_single_step_strategy(iterator, num_grad_accumulates):
            """Performs a distributed training step.

            Args:
              iterator: the distributed iterator of training datasets.

            Raises:
              ValueError: Any of the arguments or tensor shapes are invalid.
            """
            if num_grad_accumulates != 1:
                for grad_idx in tf.range(num_grad_accumulates):
                    strategy.run(_forward, args=(next(iterator),))
                    if (grad_idx + 1) % num_grad_accumulates == 0:
                        strategy.run(_step, args=(num_grad_accumulates,))
            else:
                strategy.run(_replicated_step, args=(next(iterator),))

        def train_single_step(iterator, num_grad_accumulates, first_batch):
            """Performs a distributed training step.

            Args:
              iterator: the distributed iterator of training datasets.

            Raises:
              ValueError: Any of the arguments or tensor shapes are invalid.
            """
            if num_grad_accumulates != 1:
                for _ in tf.range(num_grad_accumulates):
                    _forward(next(iterator))
                    if (_ + 1) % num_grad_accumulates == 0:
                        _step(num_grad_accumulates)
                    if hvd and _ == 0 and first_batch:
                        hvd.broadcast_variables(model.variables, 0)
                        hvd.broadcast_variables(optimizer.variables(), 0)
            else:
                _replicated_step(next(iterator), first_batch)

        def test_step(iterator):
            """Calculates evaluation metrics on distributed devices."""

            def _test_step_fn(inputs):
                """Replicated accuracy calculation."""

                inputs, labels = inputs
                model_outputs = model(inputs, is_training=False)
                run_metric_fn(config, eval_metrics, model_outputs)

            if strategy:
                strategy.run(_test_step_fn, args=(next(iterator),))
            else:
                _test_step_fn(next(iterator))

        if not run_eagerly:
            train_single_step = tf.function(train_single_step)
            test_step = tf.function(test_step)

        def _run_evaluation(current_training_step, test_iterator):
            """Runs validation steps and aggregate metrics."""
            for _ in range(eval_steps):
                test_step(test_iterator)

            with eval_summary_writer.as_default():
                for key, metric in eval_metrics.items():
                    metric_value = _float_metric_value(metric)
                    log(
                        "Step: [{}] Validation {} = {}".format(
                            current_training_step,
                            key,
                            metric_value,
                        )
                    )
                    tf.summary.scalar(key, metric_value, step=current_training_step)
                eval_summary_writer.flush()

        def _run_callbacks_on_batch_begin(batch):
            """Runs custom callbacks at the start of every step."""
            if not custom_callbacks:
                return
            for callback in custom_callbacks:
                callback.on_batch_begin(batch)

        def _run_callbacks_on_batch_end(batch):
            """Runs custom callbacks at the end of every step."""
            if not custom_callbacks:
                return
            for callback in custom_callbacks:
                callback.on_batch_end(batch)

        # Training loop starts here.
        # Set up model checkpoint
        checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0),
            phase2=tf.Variable(False),
            optimizer=optimizer,
            model=model,
        )

        manager = tf.train.CheckpointManager(
            checkpoint, config.checkpoints_dir, max_to_keep=config.keep_checkpoint_max
        )

        if config.restore_checkpoint and config.restore_checkpoint != "latest":
            checkpoint.restore(config.restore_checkpoint)
            log(
                " ** Restored model checkpoint from {}".format(
                    config.restore_checkpoint
                )
            )
        elif (
            config.restore_checkpoint
            and config.restore_checkpoint == "latest"
            and manager.latest_checkpoint
        ):
            checkpoint.restore(manager.latest_checkpoint)
            log(
                " ** Restored model checkpoint from {}".format(
                    manager.latest_checkpoint
                )
            )
        elif config.load_weights:
            model.generator(model.generator.dummy_inputs)
            model.discriminator(model.discriminator.dummy_inputs)
            model.generator.load_weights(
                os.path.join(config.weights_dir, "generator", "tf_model.h5")
            )
            model.discriminator.load_weights(
                os.path.join(config.weights_dir, "discriminator", "tf_model.h5")
            )
        else:
            log(" ** Initializing from scratch.")

        restore_iterator = (
            bool(config.restore_checkpoint) and config.restore_checkpoint == "latest"
        )

        # Initialize global step for phase2
        if config.phase2 and not bool(checkpoint.phase2):
            optimizer.iterations.assign(0)
            checkpoint.step.assign(0)
            checkpoint.phase2.assign(True)
            restore_iterator = False
        if bool(checkpoint.phase2):
            manager = tf.train.CheckpointManager(
                checkpoint,
                config.checkpoints_dir,
                checkpoint_name="ckpt-p2",
                max_to_keep=config.keep_checkpoint_max,
            )

        if hvd:
            # Set up iterator checkpoint
            iter_checkpoint = tf.train.Checkpoint(
                train_iterator=train_iterator,
                world_size=tf.Variable(hvd.size()),
                rank=tf.Variable(hvd.rank()),
            )

            iter_manager = tf.train.CheckpointManager(
                iter_checkpoint,
                os.path.join(
                    config.checkpoints_dir,
                    "iter_ckpt_rank_" + "{:02}".format(hvd.rank()),
                ),
                checkpoint_name="iter_ckpt_rank_" + "{:02}".format(hvd.rank()),
                max_to_keep=config.keep_checkpoint_max,
            )

            if restore_iterator and iter_manager.latest_checkpoint:
                ckpt_world_size = tf.train.load_variable(
                    iter_manager.latest_checkpoint,
                    "world_size/.ATTRIBUTES/VARIABLE_VALUE",
                )
                if ckpt_world_size == hvd.size():
                    iter_checkpoint.restore(iter_manager.latest_checkpoint)
                    log(
                        " ** Restored iterator checkpoint from {}".format(
                            iter_manager.latest_checkpoint
                        )
                    )

        # log(" *********** CHECKING EMBEDDINGS ***********")
        # log(model.generator.deberta.embeddings.weight)
        # log(model.discriminator.debertav2.deberta.embeddings.weight)

        current_step = int(checkpoint.step.numpy())

        checkpoint_name = "ckpt-{step}.ckpt"

        utils.heading("Running training")
        start_time = time.time()
        first_steps = current_step
        total_running_steps = total_training_steps - first_steps
        global_batch_size = config.train_batch_size * num_accumulative_step
        if hvd:
            global_batch_size *= hvd.size()

        log(" ** Starting training from step {}".format(current_step))
        log(" ** Remaining steps {}".format(total_running_steps))
        train_start = time.time()
        accum_gradients.reset()
        while current_step < total_training_steps:
            # Training loss/metric are taking average over steps inside micro
            # training loop. We reset the their values before each round.
            train_loss_metric.reset_states()
            for metric in train_metrics.values():
                metric.reset_states()

            _run_callbacks_on_batch_begin(current_step)
            # Runs several steps in the host while loop.
            # steps = steps_to_run(current_step, eval_every_n_steps, steps_per_loop)
            steps = config.log_freq

            t0_wo = time.time()
            if steps == 1:
                # TODO(zongweiz): merge with train_steps once tf.while_loop
                # GPU performance bugs are fixed.
                if strategy:
                    train_single_step_strategy(train_iterator, num_accumulative_step)
                else:
                    train_single_step(
                        train_iterator, num_accumulative_step, first_batch
                    )
            else:
                # Converts steps to a Tensor to avoid tf.function retracing.
                if strategy:
                    train_steps_strategy(
                        train_iterator,
                        tf.convert_to_tensor(steps, dtype=tf.int32),
                        num_accumulative_step,
                    )
                else:
                    train_steps(
                        train_iterator,
                        tf.convert_to_tensor(steps, dtype=tf.int32),
                        num_accumulative_step,
                        first_batch,
                    )

            if first_batch and (not hvd or hvd.rank() == 0):
                log(model.summary())

            first_batch = False
            _run_callbacks_on_batch_end(current_step)
            current_step += steps

            train_loss = _float_metric_value(train_loss_metric)

            train_metrics["train_perf"].update_state(
                steps * global_batch_size / (time.time() - t0_wo)
            )

            train_metrics["learning_rate"].update_state(
                optimizer._optimizer._decayed_lr("float32")
            )
            train_metrics["loss_scale"].update_state(
                optimizer.loss_scale if config.amp else 1
            )

            # log(" *********** CHECKING EMBEDDINGS Again***********")
            # log(model.generator.deberta.embeddings.weight)
            # log(model.discriminator.debertav2.deberta.embeddings.weight)
            # log(model.discriminator.debertav2.deberta.embeddings.delta_embeds)

            # Updates training logging.
            log_info_dict = {
                k: float(v.result().numpy() * 100)
                if "accuracy" in k
                else float(v.result().numpy())
                for k, v in train_metrics.items()
            }

            training_status = """
            Step:{step:6d} out of {total_training_steps:6d},
            Loss:{total_loss:10.6f},
            Gen_loss:{masked_lm_loss:10.6f},
            Gen_acc:{masked_lm_accuracy:6.2f},
            Perf:{train_perf:4.0f},
            Loss Scaler: {loss_scale:4.0f},
            Learning Rate: {learning_rate:.6f},
            Elapsed: {elapsed},
            ETA: {eta},
            """
            +"""
            Disc_loss:{disc_loss:10.6f},
            Disc_acc:{disc_accuracy:6.2f},
            """ if "disc_loss" in log_info_dict else ""

            training_status = training_status.format(
                step=current_step,
                total_training_steps=total_training_steps,
                **log_info_dict,
                elapsed=utils.get_readable_time(time.time() - train_start),
                eta=utils.get_readable_time(
                    (time.time() - train_start)
                    / (current_step - first_steps)
                    * (config.num_train_steps - current_step)
                ),
            )

            if not hvd or hvd.rank() == 0:
                log(training_status)

            checkpoint.step.assign(int(optimizer.iterations))

            # save_checkpoint
            if (
                config.save_checkpoints_steps > 0
                and current_step % config.save_checkpoints_steps == 0
            ):

                if not hvd or hvd.rank() == 0:
                    save_path = manager.save(checkpoint_number=current_step)
                    log("Saved checkpoint to {}".format(save_path))

                if hvd:
                    iter_save_path = iter_manager.save(checkpoint_number=current_step)
                    log(
                        " ** Saved iterator checkpoint for step {}: {}".format(
                            current_step, iter_save_path
                        )
                    )

            if train_summary_writer:
                with train_summary_writer.as_default():
                    tf.summary.scalar(
                        train_loss_metric.name, train_loss, step=current_step
                    )
                    for key, metric in train_metrics.items():
                        metric_value = _float_metric_value(metric)
                        tf.summary.scalar(key, metric_value, step=current_step)
                    train_summary_writer.flush()

            # Saves model checkpoints and run validation steps at every epoch end.
            if eval_every_n_steps > 0 and current_step % eval_every_n_steps == 0:
                # To avoid repeated model saving, we do not save after the last
                # step of training.
                if current_step < total_training_steps and (not hvd or hvd.rank() == 0):
                    manager.save()

                if eval_input_fn:
                    log("Running evaluation after step: {}}.".format(current_step))
                    _run_evaluation(
                        current_step, _get_input_iterator(eval_input_fn, strategy)
                    )
                    # Re-initialize evaluation metric.
                    for metric in eval_metrics.values():
                        metric.reset_states()

        total_time = time.time() - start_time
        if not hvd or hvd.rank() == 0:
            save_path = manager.save(checkpoint_number=current_step)
            log("Saved checkpoint to {}".format(save_path))

            if eval_input_fn:
                log("Running final evaluation after training is complete.")
                _run_evaluation(
                    current_step, _get_input_iterator(eval_input_fn, strategy)
                )

            training_summary = {
                "total_training_steps": total_training_steps,
                "train_loss": _float_metric_value(train_loss_metric),
            }
            if eval_metrics:
                # TODO(hongkuny): Cleans up summary reporting in text.
                for key, metric in train_metrics.items():
                    training_summary[f"last_train_metrics_{key}"] = _float_metric_value(
                        metric
                    )

                for key, metric in eval_metrics.items():
                    training_summary[f"eval_metrics_{key}"] = _float_metric_value(
                        metric
                    )

            write_txt_summary(training_summary, summary_dir)

            total_sentences = total_training_steps * global_batch_size
            log("-----------------------------")
            log("  Batch size = {}".format(config.train_batch_size))
            log("  Num steps = {}".format(total_training_steps))
            log("  LR = {}".format(config.learning_rate))
            if hvd:
                log("Multi-GPU training with TF Horovod")
                log("hvd.size() = {}".format(hvd.size()))
            log(
                "Total Training Time = {:.0.2f} for Sentences = {}".format(
                    total_time,
                    total_sentences,
                )
            )
            if total_time != 0:
                log(
                    "Throughput Average (sentences/sec) with overhead = {:.0.2f}".format(
                        total_sentences / total_time
                    )
                )
            log("-----------------------------")

        return model
