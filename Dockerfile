FROM continuumio/miniconda3

WORKDIR /app

RUN conda update -n base -c defaults conda

RUN conda create -n deephit python=2.7

COPY install.sh ./install.sh

SHELL ["conda", "run", "-n", "deephit", "/bin/bash", "-c"]

RUN bash install.sh

COPY . ./

ENTRYPOINT ["conda", "run", "-n", "deephit", "python", "deephit.py"]
