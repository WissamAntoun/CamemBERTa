import argparse
import collections
import json
import os
import shutil

import numpy as np
from transformers import AutoModel, AutoModelForMaskedLM

TOKENIZER_FILES_NAMES = [
    "added_tokens.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "spm.model",
    "special_tokens_map.json",
]


def convert_tf_ckpt(args):
    print("Converting TF checkpoint to PT checkpoint")
    print("Loading TF checkpoint from {}".format(args.tf_checkpoint_dir))
    if args.is_mlm:
        print("Loading MLM model")
        model = AutoModelForMaskedLM.from_pretrained(
            args.tf_checkpoint_dir, from_tf=True
        )
    else:
        print("Loading Non-MLM model")
        model = AutoModel.from_pretrained(
            args.tf_checkpoint_dir,
            from_tf=True,
        )
    output_dir = (
        args.pt_checkpoint_dir
        if args.pt_checkpoint_dir
        else os.path.join(args.tf_checkpoint_dir, "pytorch_model")
    )
    print("Saving PT checkpoint to {}".format(output_dir))
    model.save_pretrained(output_dir)

    for files in os.listdir(args.tf_checkpoint_dir):
        if files in TOKENIZER_FILES_NAMES:
            print("Copying {}".format(files))
            shutil.copy(os.path.join(args.tf_checkpoint_dir, files), output_dir)


if __name__ == "__main__":
    # Parse essential args
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_checkpoint_dir", required=True, type=str)
    parser.add_argument("--is_mlm", type=bool, default=False)
    parser.add_argument("--pt_checkpoint_dir", type=str)
    args = parser.parse_args()

    convert_tf_ckpt(args)
