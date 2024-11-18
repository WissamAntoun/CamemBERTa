import argparse
import glob
import multiprocessing as mp
import os
import random
from multiprocessing import Pool
from typing import Any, Dict, List

import nltk
import pyarrow.parquet as pq
import tensorflow as tf
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    CamembertTokenizerFast,
    DebertaV2Tokenizer,
    DebertaV2TokenizerFast,
)
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
        self._current_sentences = []
        self._current_length = 0
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

        if (
            first_segment
            and (len(first_segment) + len(second_segment)) >= self._target_length / 2
        ):
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
            if len(first_segment) + len(second_segment) > 60:
                examples.append(self._make_tf_example(first_segment, second_segment))

        # small chance for random-length instead of max_length-length example
        if random.random() < 0.05:
            self._target_length = random.randint(61, self._max_length)
        else:
            self._target_length = self._max_length

        return examples

    def _make_tf_example(self, first_segment, second_segment):
        """Converts two "segments" of text into a tf.train.Example."""

        input_ids = [self.cls_token_id] + first_segment + [self.sep_token_id]
        segment_ids = [0] * len(input_ids)
        if second_segment and len(second_segment) > 30:
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

        pbar = tqdm(
            desc=f"Processing Documents - Rank {self._job_id}",
            unit="doc",
        )
        for document in batch_text:
            examples = self._example_builder.get_examples(
                document[self.dataset_text_field]
            )
            if examples:
                for example in examples:
                    self._writers[self.n_written % len(self._writers)].write(
                        example.SerializeToString()
                    )
                    self.n_written += 1
            pbar.update(1)

        return self.n_written

    def finish(self):
        for writer in self._writers:
            writer.close()


class CulturaXReader(object):
    def __init__(self, data_dir, rank, num_jobs):
        self.data_dir = data_dir
        self.rank = rank
        self.num_jobs = num_jobs

        # In the CamemBERT 2.0 paper, we only used the first 256 files enough because
        # compute resources were limited for training on the full dataset with long sequences.
        # self.all_files = glob.glob(os.path.join(self.data_dir, "*.parquet"))[:256]
        self.all_files = glob.glob(os.path.join(self.data_dir, "*.parquet"))
        self.selected_files = []
        for i, file in enumerate(self.all_files):
            if i % self.num_jobs == self.rank:
                self.selected_files.append(file)

    def __iter__(self):
        return self.__next__()

    def __next__(self):
        for file in self.selected_files:
            with open(file, "rb") as f:
                pf = pq.ParquetFile(f)
                for group_i in range(pf.num_row_groups):
                    for row in pf.read_row_group(group_i).to_pylist():
                        yield row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--dataset_text_field", type=str, default="text")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--output_filename", type=str, default="pretrain_data")
    parser.add_argument("--n_training_shards", type=int, default=100)
    parser.add_argument("--n_processes", type=int, default=8)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--double_sep", action="store_true")
    parser.add_argument("--constant_segment_ids", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()

    mkdir(args.output_dir)

    log("Start: Create Pretraining Files")
    example_writer = ExampleWriter(
        job_id=args.rank,
        vocab_file=args.tokenizer_path,
        dataset_text_field=args.dataset_text_field,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        num_jobs=args.n_processes,
        do_lower_case=False,
        output_name_prefix=args.output_filename,
        num_out_files=args.n_training_shards,
        double_sep=args.double_sep,
        constant_segment_ids=args.constant_segment_ids,
    )
    reader = CulturaXReader(
        data_dir=args.dataset_dir, rank=args.rank, num_jobs=args.n_processes
    )
    total = example_writer.write_examples(reader)
    example_writer.finish()

    print("Total examples written: {}".format(total))
    log("End: Create Pretraining Files")

# python build_pretraining_dataset_culturax_long.py \
# --dataset_dir=/scratch/data/CulturaX/fr/ \
# --dataset_text_field=text \
# --output_dir=/scratch/camembertv2/data/tfrecords/tfrecord_lower_case_0_seq_len_1024_random_seed_12345/culturax/train \
# --output_filename=culturax_data \
# --n_training_shards=10000 \
# --n_processes=513 \
# --tokenizer_path=<TOKENIZER_PATH> \
# --max_seq_length=1024 \
# --double_sep \
# --constant_segment_ids \
# --rank=$RANK
