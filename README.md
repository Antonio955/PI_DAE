>📋  A template README.md for code accompanying a Machine Learning paper

# Opening the Black Box: Illuminating energy data imputation with building physics insights

This repository is the official implementation of [Opening the Black Box: Illuminating energy data imputation with building physics insights](link to paper). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install Keras==2.2.4
pip install optuna==1.5.0
pip install pickle5==0.0.12
pip install scipy==1.5.4
pip install tensorflow==1.14.0
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Data

Download the full dataset from <https://bbd.labworks.org/ds/bbd/lbnlbldg59>

>📋  Describe where to download the data

## Preprocessing

To preprocess the data, run this command (use --help for further information):

```preprocessing
python Codes/processing.py --input_directory Data/lbnlbldg59/ --output_data Data/lbnlbldg59/processed/dataset_new.csv
```

>📋  Describe how to preprocess data

## Day-to-day matrix

Create 10 random shuffled day-to-day matrices, by running this command (use --help for further information):

```matrix creation
python Codes/create_matrices.py --input_data Data/lbnlbldg59/processed/dataset_new.csv --output_directory Data/lbnlbldg59/processed/shuffled_data/ --seeds 1
```

>📋  Describe how to create matrices

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
