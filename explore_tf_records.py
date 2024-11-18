import argparse

import tensorflow as tf
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--max_seq_length", type=int, default=512)

args = parser.parse_args()

max_seq_length = args.max_seq_length

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

name_to_features = {
    "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
    "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
    "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
}


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    return example


dataset = tf.data.TFRecordDataset(args.input_file)
dataset = dataset.map(lambda record: _decode_record(record, name_to_features))

dataset_iter = iter(dataset)

print(tokenizer.convert_ids_to_tokens(next(dataset_iter)["input_ids"].numpy()))
