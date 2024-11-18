# CamemBERTa: A French language model based on DeBERTa V3

The repos contains the code for training CamemBERTa, CamemBERTv2, and CamemBERTav2, a set of French language models based on DeBERTa V3, a DeBerta V2 with ELECTRA style pretraining using the Replaced Token Detection (RTD) objective, and on RoBERTa with the Masked Language Model (MLM) objective.
RTD uses a generator model, trained using the MLM objective, to replace masked tokens with plausible candidates, and a discriminator model trained to detect which tokens were replaced by the generator.
Usually the generator and discriminator share the same embedding matrix, but the authors of DeBERTa V3 propose a new technique to disentagle the gradients of the shared embedding between the generator and discriminator called gradient-disentangled embedding sharing (GDES)
This the first publicly available implementation of DeBERTa V3, and the first publicly DeBERTaV3 model outside of the original [Microsoft release](https://github.com/microsoft/DeBERTa) .

CamemBERT 2.0 Paper: https://arxiv.org/pdf/2411.08868

CamemBERTa Paper: https://aclanthology.org/2023.findings-acl.320/

Models:
- [CamemBERTa](https://huggingface.co/almanach/camemberta-base)
- [CamemBERTa Generator](https://huggingface.co/almanach/camemberta-base-generator)
- [CamemBERTaV2](https://huggingface.co/almanach/camembertav2-base)
- [CamemBERTv2](https://huggingface.co/almanach/camembertv2-base)

## Model update details

The new v2 update includes:
- Much larger pretraining dataset: 275B unique tokens (previously ~32B)
- A newly built tokenizer based on WordPiece with 32,768 tokens, addition of the newline and tab characters, support emojis, and better handling of numbers (numbers are split into two digits tokens)
- Extended context window of 1024 tokens

More details are available in the [CamemBERTv2 paper](https://arxiv.org/abs/2411.08868).

## Gradient-Disentangled Embedding Sharing (GDES)

To disentagle the gradients of the shared embedding between the generator and discriminator, the authors of DeBERTaV3 make use of an another embedding layer that is not shared between the generator and discriminator.
This layers is initialized to zero and added to a copy of the generator embedding matrix with diabled gradients, and should encode the difference between the generator embedding and the discriminator embedding, in order to stop the tug-of-war between the two models in the ELECTRA objective.
When training ends, the final embedding matrix of the discriminator is the sum of the generator embedding matrix and the disentangled embedding matrix.

The code for GDES is added in the [`TFDebertaV3Embeddings`](https://gitlab.inria.fr/almanach/CamemBERTa/-/blob/main/modeling_tf_deberta_v2.py#L1143) class, with the stop gradient operation added [here](https://gitlab.inria.fr/almanach/CamemBERTa/-/blob/main/modeling_tf_deberta_v2.py#L1183).
The embedding sharing is added in the [`PretrainingModel`](https://gitlab.inria.fr/almanach/CamemBERTa/-/blob/main/modeling_tf_deberta_v2.py#L2288) class initialization.

## Pretraining Setup

The v2 models were trained on French OSCAR dumps from the CulturaX Project, French scientific documents from HALvest, and the French Wikipedia for a total of 275B tokens.[CamemBERTav2-Collections](https://huggingface.co/collections/almanach/camembertav2-finetunes-6736601c501abd86ce3a0ef6) and [CamemBERTv2-Collections](https://huggingface.co/collections/almanach/camembertv2-finetunes-67365f0b0c6b2cc06829cb3c)

The v1 model was trained on the French subset of the CCNet corpus (the same subset used in CamemBERT and PaGNOL) and is available on the HuggingFace model hub: [CamemBERTa](https://huggingface.co/almanach/camemberta-base) and [CamemBERTa Generator](https://huggingface.co/almanach/camemberta-base-generator).

To speed up the pre-training experiments, the pre-training was split into two phases:

- For the v2 models:
in phase 1, the model is trained with a maximum sequence length of 512 tokens for 91,500/270,000 steps for CamemBERTav2/CamemBERTv2 with 10,000 warm-up steps and a large batch size of 8192.
In phase 2, maximum sequence length is increased to the full model capacity of 1024 tokens for 17,000 steps with 1000 warm-up steps and a batch size of 8192.

- For the v1 model:
in phase 1, the model is trained with a maximum sequence length of 128 tokens for 10,000 steps with 2,000 warm-up steps and a very large batch size of 67,584.
In phase 2, maximum sequence length is increased to the full model capacity of 512 tokens for 3,300 steps with 200 warm-up steps and a batch size of 27,648.

The model would have seen 133B tokens compared to 419B tokens for CamemBERT-CCNet which was trained for 100K steps, this represents roughly 30% of CamemBERT’s full training.
To have a fair comparison, we trained a RoBERTa model, CamemBERT30%, using the same exact pretraining setup but with the MLM objective.


### Pretraining Loss Curves

check the huggingface repo for the tensorboard logs and plots

## Fine-tuning results

Datasets: POS tagging and Dependency Parsing (GSD, Rhapsodie, Sequoia, FSMB), NER (FTB), the FLUE benchmark (XNLI, CLS, PAWS-X), and the French Question Answering Dataset (FQuAD)

## Fine-tuning Results:

Datasets: POS tagging and Dependency Parsing (GSD, Rhapsodie, Sequoia, FSMB), NER (FTB), the FLUE benchmark (XNLI, CLS, PAWS-X), the French Question Answering Dataset (FQuAD), Social Media NER (Counter-NER), and Medical NER (CAS1, CAS2, E3C, EMEA, MEDLINE).

| Model             | UPOS      | LAS       | FTB-NER   | CLS       | PAWS-X    | XNLI      | F1 (FQuAD) | EM (FQuAD) | Counter-NER | Medical-NER |
|-------------------|-----------|-----------|-----------|-----------|-----------|-----------|------------|------------|-------------|-------------|
| CamemBERT         | 97.59     | 88.69     | 89.97     | 94.62     | 91.36     | 81.95     | 80.98      | 62.51      | 84.18       | 70.96       |
| CamemBERTa        | 97.57     | 88.55     | 90.33     | 94.92     | 91.67     | 82.00     | 81.15      | 62.01      | 87.37       | 71.86       |
| CamemBERT-bio     | -         | -         | -         | -         | -         | -         | -          | -          | -           | 73.96       |
| CamemBERTv2       | 97.66     | 88.64     | 81.99     | 95.07     | 92.00     | 81.75     | 80.98      | 61.35      | 87.46       | 72.77       |
| **CamemBERTav2**  | **97.71** | 88.65     | **93.40** | **95.63** | **93.06** | **84.82** | **83.04**  | **64.29**  | **89.53**   | **73.98**   |

The following table compares CamemBERTa's performance on XNLI against other models under different training setups, which demonstrates the data efficiency of CamelBERTa.


| Model             | XNLI (Acc.) | Training Steps | Tokens seen in pre-training | Dataset Size in Tokens |
|-------------------|-------------|----------------|-----------------------------|------------------------|
| mDeBERTa          | 84.4        | 500k           | 2T                          | 2.5T                   |
| CamemBERTa        | 82.0        | 33k            | 0.139T                      | 0.031T                 |
| CamemBERTav2      | **84.82**   | 108k           | 0.4T                        | 0.275T                 |
| XLM-R             | 81.4        | 1.5M           | 6T                          | 2.5T                   |
| CamemBERT - CCNet | 81.95       | 100k           | 0.419T                      | 0.031T                 |
| CamemBERTv2       | 81.75       | 285k           | 1T                          | 0.275T                 |

*Note: The CamemBERTa training steps was adjusted for a batch size of 8192.*

## How to use CamemBERTa

Our pretrained weights are available on the HuggingFace model hub, you can load them using the following code:

```python
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

camemberta = AutoModel.from_pretrained("almanach/camemberta-base")
tokenizer = AutoTokenizer.from_pretrained("almanach/camemberta-base")

camemberta_gen = AutoModelForMaskedLM.from_pretrained("almanach/camemberta-base-generator")
tokenizer_gen = AutoTokenizer.from_pretrained("almanach/camemberta-base-generator")
```

We also include the TF2 weights including the weights for the model's RTD head for the discriminator, and the MLM head for the generator.

CamemBERTa is compatible with most finetuning scripts from the `transformers` library.

## Features:

- XLA support
- FP16 support
- BF16 support
- Horovod support
- Tensorflow Strategy support
- TPU support
- Customizable Generator depth and width
- Export to PyTorch
- Relatively easy extension to other models
- Profiling support
- Recording Gradients support

## Data Preparation

The data prep code is a verbatim copy from NVIDA ELECTRA TF2.

Following NVidia, we recommend using Docker/Singularity for setting up the environment and training.

## Pretraining

We used Singularity for training on a HPC cluster running OAR (not SLURM), but it should go something like this if you are doing local pretraining. Check the `configuration_deberta_v2.py` file and the `configs` folder for the configuration options.

SLURM users need to add `tf.distribute.cluster_resolver.SlurmClusterResolver` to the `official_utils.misc.distribution_utils.get_distribution_strategy()` function. You can also check the [NVidia BERT TF2](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow2/LanguageModeling/BERT) repository for more advanced ways to run the pretraining.

```bash
python run_pretraining --config_file=configs/p1_local_1gpu.json
```

## Finetuning

We suggest post processing the model using `postprocess_pretrained_ckpt.py` and `convert_to_pt.py` script, and then using PyTorch and HuggingFace's `transformers` library to finetune the model.

Note: After conversion check if the model `config.json` and the tokenizer configs are correct.

## Notes:

Treat this as research code, you might find some small bugs, so be patient.

- When pretraining from scratch Verify that the IDs of the control tokens are correct in the config classes.

- This code base is mostly based on the [NVidia ELECTRA TF2](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow2/LanguageModeling/ELECTRA) implementation for the ELECTRA objective, the [NVidia BERT TF2](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow2/LanguageModeling/BERT) implementation for the TF2 training loop that supports horovod as well as TF2 strategy training, and the Deberta v2 implementation from huggingface transformers.

- Changed the pretraining code to only accept config files instead of the annoying way that the original implementation did :).

- We verified that Horovod works in multi-node mode, but only verified that multi-gpu works with TF2 strategy (the logging code might break in the strategy impl. at least).

- Training with TPU runs but is super slow, check this issue for more info https://github.com/huggingface/transformers/issues/18239. (Also you will have to manually disable some logging lines since they cause some issues with the TPU).

- The repo also support training with MLM or ELECTRA, with DebertaV2 and RoBERTa. The pretraining code could be improved to support other models, by doing some abstractions to make it easier to add support for other models.

- Training (finetuning) DeBERTa V2 is ~30% slower than RoBERTa or BERT models even with XLA and FP16 or BF16.


## License

This code is licensed under the Apache License 2.0. The public model weights are licensed under MIT License.

## Citation

CamemBERT(a)-v2 paper under review:

You can use the preprint citation for now:

```bibtex
@misc{antoun2024camembert20smarterfrench,
      title={CamemBERT 2.0: A Smarter French Language Model Aged to Perfection},
      author={Wissam Antoun and Francis Kulumba and Rian Touchent and Éric de la Clergerie and Benoît Sagot and Djamé Seddah},
      year={2024},
      eprint={2411.08868},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.08868},
}
```

CamemBERTa paper accepted to Findings of ACL 2023.


```bibtex
@inproceedings{antoun-etal-2023-data,
    title = "Data-Efficient {F}rench Language Modeling with {C}amem{BERT}a",
    author = "Antoun, Wissam  and
      Sagot, Beno{\^\i}t  and
      Seddah, Djam{\'e}",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.320",
    doi = "10.18653/v1/2023.findings-acl.320",
    pages = "5174--5185"
}
```

## Contact

Wissam Antoun: `wissam (dot) antoun (at) inria (dot) fr`

Benoit Sagot: `benoit (dot) sagot (at) inria (dot) fr`

Djame Seddah: `djame (dot) seddah (at) inria (dot) fr`
