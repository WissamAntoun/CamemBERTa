# CamemBERTa: A French language model based on DeBERTa V3

The repos contains the code for training CamemBERTa, a French language model based on DeBERTa V3, a DeBerta V2 with ELECTRA style pretraining, with gradient-disentangled embedding sharing (GDES) added between the generator and discriminator.
This the first publicly available implementation of DeBERTa V3, and the first publicly DeBERTaV3 model outside of the original [Microsoft release](https://github.com/microsoft/DeBERTa) .

## Gradient-Disentangled Embedding Sharing (GDES)

To disentagle the gradients of the shared embedding between the generator and discriminator, the authors make use of an another embedding layer that is not shared between the generator and discriminator. This layers is initialized to zero and added to a copy of the generator embedding matrix with diabled gradients, and should encode the difference between the generator embedding and the discriminator embedding, in order to stop the tug-of-war between the two models in the ELECTRA objective.
When training ends, the final embedding matrix of the discriminator is the sum of the generator embedding matrix and the disentangled embedding matrix.

The code for GDES is added in the [`TFDebertaV3Embeddings`](https://gitlab.inria.fr/almanach/CamemBERTa/-/blob/main/modeling_tf_deberta_v2.py#L1143) class, with the stop gradient operation added [here](https://gitlab.inria.fr/almanach/CamemBERTa/-/blob/main/modeling_tf_deberta_v2.py#L1183).
The embedding sharing is added in the [`PretrainingModel`](https://gitlab.inria.fr/almanach/CamemBERTa/-/blob/main/modeling_tf_deberta_v2.py#L2288) class initialization.

## Pretraining Setup

The model was trained on the French subset of the CCNet corpus (the same subset used in CamemBERT and PaGNOL) and is available on the HuggingFace model hub: [CamemBERTa](https://huggingface.co/almanach/camemberta-base-ccnet) and [CamemBERTa Generator](https://huggingface.co/almanach/camemberta-base-ccnet-generator).

To speed up the pre-training experiments, the pre-training was split into two phases;
in phase 1, the model is trained with a maximum sequence length of 128 tokens for 10,000 steps with 2,000 warm-up steps and a very large batch size of 67,584.
In phase 2, maximum sequence length is increased to the full model capacity of 512 tokens for 3,300 steps with 200 warm-up steps and a batch size of 27,648.

The model would have seen 133B tokens compared to 419B tokens for CamemBERT-CCNet which was trained for 100K steps, this represents roughly 30% of CamemBERT’s full training.
To have a fair comparison, we trained a RoBERTa model, CamemBERT30%, using the same exact pretraining setup but with the MLM objective.
### Pretraining Loss Curves

check the huggingface repo for the tensorboard logs and plots

## Fine-tuning results

Datasets: POS tagging and Dependency Parsing (GSD, Rhapsodie, Sequoia, FSMB), NER (FTB), the FLUE benchmark (XNLI, PAWS-X), and the French Question Answering Dataset (FQuAD)

| Model             | UPOS      | LAS       | NER       | CLS       | PAWS-X    | XNLI      | F1 (FQuAD) | EM (FQuAD) |
|-------------------|-----------|-----------|-----------|-----------|-----------|-----------|------------|------------|
| CamemBERT (CCNet) | **97.59** | **88.69** | 89.97     | 94.62     | 91.36     | 81.95     | 80.98      | **62.51**  |
| CamemBERT (30%)   | 97.53     | 87.98     | **91.04** | 93.28     | 88.94     | 79.89     | 75.14      | 56.19      |
| CamemBERTa        | 97.57     | 88.55     | 90.33     | **94.92** | **91.67** | **82.00** | **81.15**  | 62.01      |

The following table compares CamemBERTa's performance on XNLI against other models under different training setups, which demonstrates the data efficiency of CamelBERTa.


| Model             | XNLI (Acc.) | Training Steps | Tokens seen in pre-training | Dataset Size in Tokens |
|-------------------|-------------|----------------|-----------------------------|------------------------|
| mDeBERTa          | 84.4        | 500k           | 2T                          | 2.5T                   |
| CamemBERTa        | 82.0        | 33k            | 0.139T                      | 0.319T                 |
| XLM-R             | 81.4        | 1.5M           | 6T                          | 2.5T                   |
| CamemBERT - CCNet | 81.95       | 100k           | 0.419T                      | 0.319T                 |

*Note: The CamemBERTa training steps was adjusted for a batch size of 8192.*

## How to fine-tune CamemBERTa

Our pretrained weights are available on the HuggingFace model hub, you can load them using the following code:

```python
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

camemberta = AutoModel.from_pretrained("almanach/camemberta-base-ccnet")
tokenizer = AutoTokenizer.from_pretrained("almanach/camemberta-base-ccnet")

camemberta_gen = AutoModelForMaskedLM.from_pretrained("almanach/camemberta-base-ccnet-generator")
tokenizer_gen = AutoTokenizer.from_pretrained("almanach/camemberta-base-ccnet-generator")
```

We also include the TF2 weights including the weights for the model's RTD head for the discriminator, and the MLM head for the generator.


## Features:

- XLA support
- FP16 support
- Horovod support
- Tensorflow Strategy support
- Customizable Generator depth and width
- Export to PyTorch
- Relatively easy extension to other models

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

- Training (finetuning) DeBERTa V2 is ~30% slower than RoBERTa or BERT models even with XLA and FP16.


## License

This code is licensed under the Apache License 2.0. The public model weights are licensed under MIT License.

## Citation

Paper under review, will update this section when the paper is published.

## Contact

Wissam Antoun: `wissam (dot) antoun (at) inria (dot) fr`

Benoit Sagot: `benoit (dot) aagot (at) inria (dot) fr`

Djame Seddah: `djame (dot) seddah (at) inria (dot) fr`
