# Opening the Black Box: Illuminating energy data imputation with building physics insights

This repository is the official implementation of [Opening the Black Box: Illuminating energy data imputation with building physics insights](link to paper). 

## Requirements

To install requirements:

```setup
pip install Keras==2.2.4
pip install optuna==1.5.0
pip install pickle5==0.0.12
pip install scipy==1.5.4
pip install tensorflow==1.14.0
```

## Data

Download the full dataset from <https://bbd.labworks.org/ds/bbd/lbnlbldg59>

## Preprocessing

To preprocess the data, run this command (use --help for further information):

```preprocessing
py Codes/processing.py --input_directory your_directory/lbnlbldg59/lbnlbldg59.processed/LBNLBLDG59/clean_Bldg59_2018to2020/clean data/ --output_data your_directory/processed/dataset_processed.csv
```

## Day-to-day matrix

Create 10 random shuffled day-to-day matrices, by running this command (use --help for further information):

```matrix creation
py Codes/create_matrices.py --input_data your_directory/processed/dataset_processed.csv --output_directory your_directory/processed/shuffled_data/ --seeds 1
```

## Correlations

To get the correlation coefficients, run this command (use --help for further information):

```Correlation coefficients
py Codes/scatterplot_print.py --input_directory your_directory/processed/shuffled_data/ --threshold_q_cool 50 --threshold_q_heat 20
```

## Training

To train the model(s) in the paper, run this command:

```train
py Codes/scatterplot_print.py --input_directory your_directory/processed/shuffled_data/ --threshold_q_cool 50 --threshold_q_heat 20
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
