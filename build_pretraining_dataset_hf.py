import argparse
import os
import random
from typing import Dict, List

import datasets
import nltk
import tensorflow as tf

# from transformers import DebertaV2Tokenizer, DebertaV2TokenizerFast
from transformers import AutoTokenizer, CamembertTokenizerFast
from utils import log

nltk.download("punkt")

random.seed(42)


def mkdir(path):
    if not tf.io.gfile.exists(path):
        tf.io.gfile.makedirs(path)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


class ExampleBuilder(object):
    """Given a stream of input text, creates pretraining examples."""

    def __init__(
        self, tokenizer, max_length, double_sep=False, constant_segment_ids=True
    ):
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._target_length = max_length

        vocab = self._tokenizer.vocab
        self.cls_token_id = vocab["[CLS]"]
        self.sep_token_id = vocab["[SEP]"]
        # RoBERTa uses [CLS] A [SEP] [SEP] B [SEP] for pretraining
        # Electra/BERT uses [CLS] A [SEP] B [SEP]
        self.double_sep = double_sep
        self.constant_segment_ids = constant_segment_ids
        self.segment_2_id = 0 if constant_segment_ids else 1

    def get_examples(self, document: str):
        # encode a long document and split it into multiple examples based on max_length
        # small chance to only have one segment as in classification tasks
        if random.random() < 0.5:
            first_segment_target_length = self._target_length
        else:
            # -4 due to not yet having [CLS]/([SEP]x3) tokens in the input text
            if self.double_sep:
                first_segment_target_length = (self._target_length - 4) // 2
            else:
                first_segment_target_length = (self._target_length - 3) // 2

        examples = []
        first_segment = []
        second_segment = []
        lines = nltk.sent_tokenize(document)
        for line in lines:
            bert_tokens = self._tokenizer.encode(line, add_special_tokens=False)
            if len(bert_tokens) == 0:
                continue
            if (
                len(first_segment) == 0
                or len(first_segment) + len(bert_tokens) < first_segment_target_length
            ) and (len(second_segment) == 0):
                first_segment += bert_tokens
            else:
                second_segment += bert_tokens

            if len(first_segment) + len(second_segment) >= self._target_length:
                # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens

                if self.double_sep:
                    first_segment = first_segment[: self._target_length - 4]
                    second_segment = second_segment[
                        : max(0, self._target_length - len(first_segment) - 4)
                    ]
                else:
                    first_segment = first_segment[: self._target_length - 3]
                    second_segment = second_segment[
                        : max(0, self._target_length - len(first_segment) - 3)
                    ]
                if len(first_segment) + len(second_segment) > 20:
                    examples.append(
                        self._make_tf_example(first_segment, second_segment)
                    )
                first_segment = []
                second_segment = []

        if first_segment:
            if self.double_sep:
                first_segment = first_segment[: self._target_length - 4]
                second_segment = second_segment[
                    : max(0, self._target_length - len(first_segment) - 4)
                ]
            else:
                first_segment = first_segment[: self._target_length - 3]
                second_segment = second_segment[
                    : max(0, self._target_length - len(first_segment) - 3)
                ]
            if len(first_segment) + len(second_segment) > 20:
                examples.append(self._make_tf_example(first_segment, second_segment))

        # small chance for random-length instead of max_length-length example
        if random.random() < 0.05:
            self._target_length = random.randint(21, self._max_length)
        else:
            self._target_length = self._max_length

        return examples

    def _make_tf_example(self, first_segment, second_segment):
        """Converts two "segments" of text into a tf.train.Example."""

        input_ids = [self.cls_token_id] + first_segment + [self.sep_token_id]
        segment_ids = [0] * len(input_ids)
        if second_segment and len(second_segment) > 10:
            if self.double_sep:
                input_ids += [self.sep_token_id] + second_segment + [self.sep_token_id]
                segment_ids += [self.segment_2_id] * (len(second_segment) + 2)
            else:
                input_ids += second_segment + [self.sep_token_id]
                segment_ids += [self.segment_2_id] * (len(second_segment) + 1)
        input_mask = [1] * len(input_ids)
        input_ids += [0] * (self._max_length - len(input_ids))
        input_mask += [0] * (self._max_length - len(input_mask))
        segment_ids += [0] * (self._max_length - len(segment_ids))
        tf_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "input_ids": create_int_feature(input_ids),
                    "input_mask": create_int_feature(input_mask),
                    "segment_ids": create_int_feature(segment_ids),
                }
            )
        )
        return tf_example


class ExampleWriter(object):
    """Writes pre-training examples to disk."""

    def __init__(
        self,
        job_id,
        vocab_file,
        dataset_text_field,
        output_dir,
        max_seq_length,
        num_jobs,
        do_lower_case,
        output_name_prefix,
        num_out_files=1000,
        double_sep=False,
        constant_segment_ids=True,
    ):
        self.dataset_text_field = dataset_text_field
        tokenizer = AutoTokenizer.from_pretrained(
            vocab_file, do_lower_case=do_lower_case
        )
        # tokenizer = ElectraTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        self._example_builder = ExampleBuilder(
            tokenizer, max_seq_length, double_sep, constant_segment_ids
        )
        self._writers = []
        for i in range(num_out_files):
            if i % num_jobs == job_id:
                output_fname = os.path.join(
                    output_dir,
                    "{}-{:05d}.tfrecord".format(output_name_prefix, i),
                )
                # delete existing file
                if tf.io.gfile.exists(output_fname):
                    tf.io.gfile.remove(output_fname)
                self._writers.append(tf.io.TFRecordWriter(output_fname))
        self.n_written = 0
        self._job_id = job_id
        self.output_name_prefix = output_name_prefix

    def write_examples(self, batch_text):
        """Writes out examples from the provided input file."""

        for document in batch_text[self.dataset_text_field]:
            examples = self._example_builder.get_examples(document)
            if examples:
                for example in examples:
                    self._writers[self.n_written % len(self._writers)].write(
                        example.SerializeToString()
                    )
                    self.n_written += 1

    def finish(self):
        for writer in self._writers:
            writer.close()


class HFCreateTFRecords(object):
    def __init__(
        self,
        dataset_name,
        dataset_config,
        dataset_text_field,
        output_dir,
        cache_dir,
        output_filename,
        n_training_shards,
        n_processes,
        tokenizer_path,
        max_seq_length=512,
        streaming=False,
        double_sep=False,
        constant_segment_ids=True,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        mkdir(self.output_dir)

        self.output_filename = output_filename

        if "wiki" in dataset_name:
            self.ds = datasets.load_dataset(
                dataset_name,
                date=self.dataset_config,
                language="fr",
                split="train",
                cache_dir=self.cache_dir,
            )
        else:
            self.ds = datasets.load_dataset(
                dataset_name,
                self.dataset_config,
                split="train",
                cache_dir=self.cache_dir,
                num_proc=n_processes,
                streaming=streaming,
            )

        self.dataset_text_field = dataset_text_field
        self.n_processes = n_processes

        assert n_training_shards > 0, "There must be at least one output shard."
        self.n_training_shards = n_training_shards

        self.output_training_identifier = "_training"
        self.output_file_extension = ".tfrecord"

        self.output_training_writers = {}

        self.tokenizer_path = tokenizer_path
        self.max_seq_length = max_seq_length
        self.double_sep = double_sep
        self.constant_segment_ids = constant_segment_ids

    def init_output_files(self):
        log("Start: Init Output Files")

        # for i in range(self.n_processes):
        #     self.output_training_writers[i] = example_writer

    def close_output_files(self):
        log("Start: Close Output Files")
        for writer in self.output_training_writers.values():
            writer.close()

    def write_to_output_files(self):
        def write_to_shards(examples, rank):
            example_writer = ExampleWriter(
                job_id=rank,
                vocab_file=self.tokenizer_path,
                dataset_text_field=self.dataset_text_field,
                output_dir=self.output_dir,
                max_seq_length=self.max_seq_length,
                num_jobs=self.n_processes,
                do_lower_case=False,
                output_name_prefix=self.output_filename,
                num_out_files=self.n_training_shards,
                double_sep=self.double_sep,
                constant_segment_ids=self.constant_segment_ids,
            )
            example_writer.write_examples(examples)
            example_writer.finish()

        log("Start: Write to Output Files")
        self.ds.map(
            function=write_to_shards,
            with_rank=True,
            num_proc=self.n_processes,
            batched=True,
            batch_size=50000,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_text_field", type=str, default="text")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_filename", type=str, default="pretrain_data")
    parser.add_argument("--n_training_shards", type=int, default=100)
    parser.add_argument("--n_processes", type=int, default=8)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--streaming", type=bool, default=False)
    parser.add_argument("--double_sep", action="store_true")
    parser.add_argument("--constant_segment_ids", action="store_true")
    args = parser.parse_args()

    hf_create_tfrecords = HFCreateTFRecords(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_text_field=args.dataset_text_field,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        output_filename=args.output_filename,
        n_training_shards=args.n_training_shards,
        n_processes=args.n_processes,
        tokenizer_path=args.tokenizer_path,
        max_seq_length=args.max_seq_length,
        streaming=args.streaming,
        double_sep=args.double_sep,
        constant_segment_ids=args.constant_segment_ids,
    )

    hf_create_tfrecords.init_output_files()
    hf_create_tfrecords.write_to_output_files()
    hf_create_tfrecords.close_output_files()

# python build_pretraining_dataset_hf.py \
# --dataset_name=olm/wikipedia \
# --dataset_config=\"20240401\" \
# --dataset_text_field=text \
# --cache_dir=/scratch/data/wikipedia/.cache \
# --output_dir=/scratch/camembertv2/data/tfrecords/tfrecord_lower_case_0_seq_len_512_random_seed_12345/wiki/train \
# --output_filename=wiki_data \
# --n_training_shards=256 \
# --n_processes=32 \
# --tokenizer_path=<TOKENIZER_PATH> \
# --max_seq_length=512 \
# --double_sep \
# --constant_segment_ids
