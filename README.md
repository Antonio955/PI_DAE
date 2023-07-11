# Opening the Black Box: Illuminating energy data imputation with building physics insights

This repository is the official implementation of [Opening the Black Box: Illuminating energy data imputation with building physics insights](link to paper). 

## Requirements

To install requirements (python 3.6.8):

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
py codes/processing.py --input_directory your_directory/lbnlbldg59/lbnlbldg59.processed/LBNLBLDG59/clean_Bldg59_2018to2020/clean data/ --output_directory your_directory/processed_data/
```

## Day-to-day matrix

Create 10 random shuffled day-to-day matrices, by running this command (use --help for further information):

```matrix creation
py codes/create_matrices.py --input_data your_directory/processed_data/dataset_processed.csv --output_directory your_directory/processed_data/shuffled_data/ --seeds 1
```

## Correlations

To get the correlation coefficients, run this command (use --help for further information):

```Correlation coefficients
py codes/scatterplot_print.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20
```

## Tuning

To tune the model(s) in the paper, run this command (use --help for further information):

```tune
py codes/tune.py --path your_directory --lambdaa 1 --features 4 --target t_ra --corr 0.2
```

Tuned hyperparameters can be accessed here: your_directory/Results/Tuning/Tuning.csv

## Training

To train the model(s) in the paper, run this command (use --help for further information):

```train
py codes/train.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20 --train_rate 0.1 --aug 80 --lambdaa 1 --features 4 --target t_ra
```

## Evaluation

To evaluate the model(s) in the paper, run these commands (use --help for further information):

```eval
py codes/evaluate.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20 --train_rate 0.1 --aug 80 --lambdaa 1 --features 4 --target t_ra
py codes/computational_req.py --path your_directory --train_rate 0.1 --lambdaa 1 --features 4 --target t_ra
py codes/LIN_train_evaluate.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20 --train_rate 0.1
py codes/KNN_train_evaluate.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20 --train_rate 0.1
```

## Pre-trained Models

You can access the pretrained models here: your_directory/results/pre_trained_models/

## Results

Complete results (including physics-based coefficients, trainable parameters, computational requirements and cluster documentation) can be accessed here: your_directory/results/Results.csv

To reproduce the plots in the paper, run these commands (use --help for further information):

```plots
py Codes/draw_days.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20 --train_rate 0.1 --aug 80 --lambdaa 1 --features 4 --target t_ra  --corr 0.2 --seeds 1
py codes/computational_curves_draw.py --path your_directory
py codes/learning_curves_avg_draw.py
py codes/learning_curves_std_draw.py
py codes/physics_coeff_draw.py
py codes/scatterplot_draw.py
```

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
