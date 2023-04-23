# Universal-Sentence-Representations
Reimplementing and analysing the paper "Supervised Learning of Universal Sentence Representations from Natural Language Inference Data"


1. Learn universal sentence representations
2. Test sentence embeddings on SentEval transfer tasks

## Requirements

USC has the following requirements:

* [Python](https://www.python.org/) - v3.9.15
* [PyTorch](https://pytorch.org/) - v1.12.1
* [datasets](https://huggingface.co/docs/datasets/index) - v2.11.0
* [Weights&Biases](https://wandb.ai/) - v0.13.2

## My code structure
Be sure to follow this structure to avoid model saving and loading issues

* data: folder where you should put glove embeddings
* saves: folder where pretrained models will be saved (make a subfolder with your model name)
* results: folder where evaluate results will be saved (make a subfolder with your model name)
* datasets.py: load, preprocess and format datasets
* models.py: encoder models and classifier model for SNLI task
* train.py: train and eval on SNLI function
* evaluate.py: evalute pretrained sentence representations on SentEval tasks
* utils.py: helper functions for custom input
* environment file for creating the environment

For installing the environment run:

1. conda env create -f environment.yml

For loading the glove word embeddings:

1. cd pretrained
2. !wget http://nlp.stanford.edu/data/glove.840B.300d.zip

For training a model run:

1. wandb login (login with wandb account)
2. modify train.py for your wandb project (just one line)
3. source activate USR1
4. !python train.py --model model_name

model_name = ['baseline', 'lstm', 'bilstm', 'poollstm']

For evaluation on SentEval run:

1. !git clone https://github.com/facebookresearch/SentEval.git
2. cd SentEval/
3. !python /content/SentEval/setup.py install
4. cd data/downstream/
5. !./get_transfer_data.bash
6. move downstream and probing folders to data folder: data (see above)
6. source activate USR1
7. potentially modify evaluation.py for the SentEval tasks you want to test
8. !python evaluate.py --model model_name --checkpoint_path checkpoint_path

## Link for pretrained models

https://drive.google.com/drive/folders/1YA_0TmxnzH6ekU7qmJdO7xQprJ0gkawN?usp=sharing

## Link for wandb plots of training runs

https://drive.google.com/drive/folders/1YRvKVltv3vAGHAnlIAUyH2SOljZ0-yy_?usp=sharing

## Main Citations used for this

SNLI dataset:

@article{bowman2015large,
  title={A large annotated corpus for learning natural language inference},
  author={Bowman, Samuel R and Angeli, Gabor and Potts, Christopher and Manning, Christopher D},
  journal={arXiv preprint arXiv:1508.05326},
  year={2015}
}

Supervised Learning of Universal Sentence Representations from Natural Language Inference Data:

@article{conneau2017supervised,
  title={Supervised learning of universal sentence representations from natural language inference data},
  author={Conneau, Alexis and Kiela, Douwe and Schwenk, Holger and Barrault, Loic and Bordes, Antoine},
  journal={arXiv preprint arXiv:1705.02364},
  year={2017}
}

SentEval:

@article{conneau2018senteval,
  title={Senteval: An evaluation toolkit for universal sentence representations},
  author={Conneau, Alexis and Kiela, Douwe},
  journal={arXiv preprint arXiv:1803.05449},
  year={2018}
}

