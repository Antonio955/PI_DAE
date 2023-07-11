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
py Codes/processing.py --input_directory your_directory/Data/lbnlbldg59/lbnlbldg59.processed/LBNLBLDG59/clean_Bldg59_2018to2020/clean data/ --output_data your_directory/Data/processed/dataset_processed.csv
```

## Day-to-day matrix

Create 10 random shuffled day-to-day matrices, by running this command (use --help for further information):

```matrix creation
py Codes/create_matrices.py --input_data your_directory/Data/processed/dataset_processed.csv --output_directory your_directory/Data/processed/shuffled_data/ --seeds 1
```

## Correlations

To get the correlation coefficients, run this command (use --help for further information):

```Correlation coefficients
py Codes/scatterplot_print.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20
```

## Tuning

To tune the model(s) in the paper, run this command (use --help for further information):

```tune
py Codes/tune.py --path your_directory --lambdaa 1 --features 4 --target t_ra --corr 0.2
```

Tuned hyperparameters can be accessed here: your_directory/Results/Tuning/Tuning.csv

## Training

To train the model(s) in the paper, run this command (use --help for further information):

```train
py Codes/train.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20 --train_rate 0.1 --aug 80 --lambdaa 1 --features 4 --target t_ra
```

## Evaluation

To evaluate the model(s) in the paper, run these commands (use --help for further information):

```eval
py Codes/evaluate.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20 --train_rate 0.1 --aug 80 --lambdaa 1 --features 4 --target t_ra
py Codes/computational_req.py --path your_directory --train_rate 0.1 --lambdaa 1 --features 4 --target t_ra
py Codes/LIN_train_evaluate.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20 --train_rate 0.1
py Codes/KNN_train_evaluate.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20 --train_rate 0.1
```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Complete results (including physics-based coefficients, trainable parameters, computational requirements and cluster documentation) can be accessed here: your_directory/Results/

To reproduce the plots in the paper, run these commands (use --help for further information):

```plots
py Codes/evaluate.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20 --train_rate 0.1 --aug 80 --lambdaa 1 --features 4 --target t_ra
py Codes/computational_req.py --path your_directory --train_rate 0.1 --lambdaa 1 --features 4 --target t_ra
py Codes/LIN_train_evaluate.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20 --train_rate 0.1
py Codes/KNN_train_evaluate.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20 --train_rate 0.1
```

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
