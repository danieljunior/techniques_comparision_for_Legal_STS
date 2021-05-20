# Techniques Comparision for Legal STS

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

## Execução 

#### Construção da imagem
- `docker build -f Dockerfile -t legal_sts .`

#### Iniciar containers

- `docker run --rm  -v ${PWD}:/app -w /app -p 8888:8888 --name legal_sts -itd legal_sts bash`

#### Acessar container
- `docker exec -it legal_sts bash`

#### Baixar dados
- Dentro da pasta do projeto executar:
    - `pip install gdown`
    - `gdown https://drive.google.com/uc?id=1lypU6NQORACvJpZKbU6JWPHOfefwYnZe && \
        unzip models.zip && \
        rm -f models.zip

        gdown https://drive.google.com/uc?id=1sLgo7DXAW-MHnibDpeWCK0anFbqZ8wWF && \
        unzip datasets.zip && \
        rm -f datasets.zip`

#### Iniciar jupyter notebook

- `jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='legalsts' &`
#### Acessar o jupyter notebook
- http://localhost:8888/lab
