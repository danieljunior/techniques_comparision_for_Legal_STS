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
- `docker build --rm -f Dockerfile -t legal_sts .`
- Or pull: `docker pull danielpsjr/legal_sts:1.0.0`

#### Containers initialize

- `docker run --rm --shm-size="2g" -v ${PWD}:/app -w /app -p 8888:8888 --name legal_sts -itd legal_sts bash`
- `docker run --rm --shm-size="2g" -v ${PWD}:/app -w /app --name legal_sts -itd danielpsjr/legal_sts:1.0.0 bash`

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
4. To run `finetunning_sim_cse.py` must update `sentence-transformers` from fresh docker image: `pip install --upgrade sentence-transformers`
5. To run the training of DiffCSE:  `https://github.com/danieljunior/PtBr-DiffCSE`
6. To use PtBr-DiffCSE:

```conda deactivate
conda conda create --name diffcse --clone base
conda activate diffcse
python -m ensurepip --default-pip
wget https://bootstrap.pypa.io/get-pip.py
pip install --upgrade transformers
```
