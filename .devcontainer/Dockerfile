
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04


ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN apt-get update && apt-get install -y \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p $CONDA_DIR \
    && rm ~/miniconda.sh


COPY ../environment.yaml .
RUN conda env create -f environment.yaml -n fractalgen


ENV PATH=/opt/conda/envs/fractalgen/bin:$PATH
RUN echo "source activate fractalgen" >> ~/.bashrc

COPY ../ /app
WORKDIR /app

CMD ["python", "demo/run.py"]
