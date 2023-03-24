# Comparing Unsupervised Approaches for Legal STS

## Techniques evaluated:
- TF-IDF
- BM-25
- LDA
- Word2Vec (Average e Weighted)
- FastText (Average e Weighted)
- Doc2Vec
- Sentence2Vec
- ELMo (Average e Weighted)
- BERT (Só pré-treinado e tunado)
- Longformer (Só pré-treinado e tunado)


## Env build

#### Docker image build
- `docker build -f Dockerfile -t legal_sts .`

#### Containers initialize

- `docker run --rm --shm-size="2g" -v ${PWD}:/app -w /app -p 8888:8888 --name legal_sts -itd legal_sts bash`

#### Access container
- `docker exec -it legal_sts bash`

## Code execution 

#### Download data
- Install gdown python lib and download all files needed:
    - `pip install gdown`
    - `gdown https://drive.google.com/uc?id=1UD-mK3K9KhuvJtRVwA_CqPE1E0Z2lAuR && \
        unzip models.zip && \
        rm -f models.zip

        gdown https://drive.google.com/uc?id=1iTXMfsmhs4w3qgkM3kIreePUxbms6zwC && \
        unzip datasets.zip && \
        rm -f datasets.zip`

#### Start jupyter notebook
- `jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='legalsts' &`
#### Access jupyter notebook
- http://localhost:8888/lab

#### Run scripts and notebooks
1. Run file `experiment_workflow.py`
2. Run notebook `notebooks/Data Analysis.ipynb`
3. Run notebook `notebooks/Data Analysis.ipynb`

