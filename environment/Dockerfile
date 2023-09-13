FROM nvidia/cuda:11.0.3-base-ubuntu20.04

## Install Python
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y python3 python3-pip

## Install basic data science libraries
# Required by the BinaPs implementation
RUN pip install numpy \
                scipy \
                pandas \
                torch \
                torchvision \
                torchaudio \
                --extra-index-url https://download.pytorch.org/whl/cu116

# Install Jupyter and IPython support
RUN pip install jupyterlab \
                ipykernel \
                ipython
RUN python3 -m ipykernel install --name=binaps

## Install R
# To enable us to run binaps/Data/Synthetic_data/generate_toy.R
RUN DEBIAN_FRONTEND=noninteractive apt -y install r-base

# Install extension dependencies
RUN pip install rich \
                surprise \
                numba

# Data visualization dependencies
RUN pip install matplotlib \
                plotly \
                tabulate \
                plotext \
                statsmodels \
                kaleido
RUN DEBIAN_FRONTEND=noninteractive apt -y install msttcorefonts

# Unit test tooling
RUN pip install pytest \
                coverage \
                nbmake
RUN pip install coverage

# Install Pedro's implementation dependencies
RUN pip install scikit-learn