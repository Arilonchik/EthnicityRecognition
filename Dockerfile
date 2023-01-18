FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import torch"

WORKDIR /workdir

COPY FER /workdir/FER
COPY Face_detector /workdir/Face_detector
COPY scripts /workdir/scripts
COPY entry.py /workdir/entry.py

RUN mkdir -p /workdir/mount/input/
RUN mkdir -p /workdir/mount/output/
ENTRYPOINT ["/bin/bash"]