# Comparing Unsupervised Approaches for Legal STS

## Techniques evaluated:
- TF-IDF
- BM-25
- LDA
- Word2Vec (Average e Weighted)
- FastText (Average e Weighted)
- Doc2Vec
- SentenceTransformer
- ELMo (Average e Weighted)
- BERT (pre-trained and fine-tuned)
- Longformer (pre-trained and fine-tuned)
- SimCSE
- DiffCSE

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

**Datasets**: https://iduffbr-my.sharepoint.com/:u:/g/personal/danieljunior_id_uff_br/EbIjA4sjrKtGhIA-7uz9Xl8BBaLZ9f1eAyEGT980WEF3Xg?e=2PFzQW

**Models**: https://huggingface.co/melll-uff

#### Run scripts and notebooks

1. Run script `experiment_workflow.py`


2. Run script `faiss_get_nns.py`


3. Run notebook `notebooks/Data Analysis.ipynb`
   1. Verify the following package and versions: `Jinja2-3.0.1` `bokeh==2.3.0` `pynndescent==0.5.8` (run pip install with `--ignore-installed llvmlite`) 
   2. Start jupyter notebook: `jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='legalsts' &`
   3. Access jupyter notebook: `http://localhost:8888/lab
   

4. To run the training/usage of PtBr-SimCSE:
```conda deactivate
conda create --name simcse --clone base
conda activate simcse
python -m ensurepip --default-pip
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
pip install importlib_metadata==6.7.0
pip install sentence-transformers==2.2.2
pip install pandas==1.2.4
python finetunning_sim_cse.py # to train
``` 

5. To run the training of PtBr-DiffCSE:  `https://github.com/danieljunior/PtBr-DiffCSE`


6. To use PtBr-DiffCSE:

```conda deactivate
conda create --name diffcse --clone base
conda activate diffcse
python -m ensurepip --default-pip
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
pip install --upgrade transformers
pip install pandas==1.2.4
```

