# Verwende ein Basis-Image mit Python und JupyterLab
FROM docker.io/jupyter/base-notebook:python-3.8

USER root

# Installiere fbprophet und andere notwendige Pakete
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    gfortran \
    && pip install --no-cache-dir prophet

USER jovyan

# Setze den Arbeitsverzeichnis
WORKDIR /home/jovyan/work

# Exponiere den JupyterLab Port
EXPOSE 8888

# Starte JupyterLab
CMD ["start-notebook.sh", "--NotebookApp.token=''"]

