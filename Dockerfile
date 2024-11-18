ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:24.06-tf2-py3
FROM ${FROM_IMAGE_NAME}
RUN apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract

RUN pip install --no-cache-dir \
    tqdm boto3 requests six ipdb h5py nltk progressbar filelock \
    git+https://github.com/NVIDIA/dllogger \
    nvidia-ml-py3==7.352.0

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt