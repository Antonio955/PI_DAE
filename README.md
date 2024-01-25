# Opening the Black Box: Towards inherently interpretable energy data imputation models using building physics insight

This repository is the official implementation of [Opening the Black Box: Towards inherently interpretable energy data imputation models using building physics insight][https://arxiv.org/abs/2311.16632] 

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

## Correlations and statistics

To get the correlation coefficients and general statistics, run these command (use --help for further information):

```Whole dataset
py codes/statistics.py --path your_directory
```

```Per training set
py codes/statistics_training.py --path your_directory
```

## Tuning

To tune the model(s) in the paper, run this command (use --help for further information):

```tune
py codes/tune.py --path your_directory --lambdaa 1 --features 4 --target t_ra --corr 0.2
```

Tuned hyperparameters can be accessed here: your_directory/Results/Tuning.csv

## Training

To train the model(s) in the paper, run this command (use --help for further information):

```train
py codes/train.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20 --train_rate 0.1 --seeds_coeff 0 --aug 80 --lambdaa 1 --features 4 --target t_ra
```

## Evaluation

To evaluate the model(s) in the paper, run these commands (use --help for further information):

```eval
py codes/evaluate.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20 --train_rate 0.1 --aug 80 --lambdaa 1 --features 4 --target t_ra
```

```eval
py codes/computational_req.py --path your_directory --train_rate 0.1 --lambdaa 1 --features 4 --target t_ra
```

```eval
py codes/LIN_train_evaluate.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20 --train_rate 0.1
```

```eval
py codes/KNN_train_evaluate.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20 --train_rate 0.1
```

## Pre-trained Models

You can access the pretrained models here: your_directory/results/pre_trained_models/

## Physics-based coefficients

To print the optimized physics-based coefficients, run this command (use --help for further information):

```eval
py codes/physics_coeff_print.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20 --seeds_coeff 0
```

## Results

Complete results (including physics-based coefficients, trainable parameters, computational requirements and cluster documentation) can be accessed here: your_directory/results/Results.xlsx

To reproduce the plots in the paper, run these commands (use --help for further information):

```plots
py codes/draw_days.py --path your_directory --threshold_q_cool 50 --threshold_q_heat 20 --train_rate 0.1 --aug 80 --lambdaa 1 --features 4 --target t_ra  --corr 0.2 --seeds 1
```

```plots
py codes/computational_curves_draw.py --path your_directory
```

```plots
py codes/learning_curves_avg_draw.py
```

```plots
py codes/learning_curves_std_draw.py
```

```plots
py codes/physics_coeff_draw.py
```

```plots
py codes/physics_coeff_draw_2.py
```

## Contributing

MIT License
