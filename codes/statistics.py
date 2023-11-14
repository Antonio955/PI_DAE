import pickle5 as pickle
import matplotlib.pyplot as plt
import support_fun
from scipy.stats import pearsonr
import argparse
import os
import numpy as np
import pandas as pd
parser = argparse.ArgumentParser()

# Don't change the seeds
seeds = 1

parser.add_argument("--path", help="Specify the path")

args = parser.parse_args()

path = args.path

dataset_dir = path+'/processed_data/shuffled_data/matrix_'+str(seeds)+'.pkl'      # directory containing data

with open(dataset_dir, 'rb') as handle:  # load preprocessed data
    dataset_ = (pickle.load(handle))  # [timestamp, t_oa, q_cool, q_heat, t_ra]

dataset_ = np.reshape(dataset_, (int(len(dataset_) / 48), 48, 5))

for threshold_q_cool in range(0, 51, 10):
    for threshold_q_heat in range(0, 51, 10):

        indice = support_fun.generate_indice_full(dataset_, threshold_q_cool=threshold_q_cool, threshold_q_heat=threshold_q_heat)

        dataset = dataset_[indice]

        if dataset.shape[0] < 10:
            continue

        timestamp_1 = dataset[:,:,0:1]
        ttX_p_1 = dataset[:,:,1:]

        timestamp_1 = np.reshape(timestamp_1, (int(len(timestamp_1) * 48), 1))
        ttX_p_1 = np.reshape(ttX_p_1, (int(len(ttX_p_1) * 48), 4))
        time = range(len(timestamp_1))

        df_1 = pd.DataFrame(np.hstack((timestamp_1,ttX_p_1)), columns = ['date','t_oa','q_cool','q_heat','t_ra'])
        df_1['date'] = pd.to_datetime(df_1['date'])
        df_1 = df_1.set_index('date')
        df_1.sort_index(inplace=True)
        df_1.reset_index(inplace=True)

        t_ra = df_1["t_ra"].tolist()
        q_cool = df_1["q_cool"].tolist()
        q_heat = df_1["q_heat"].tolist()
        t_oa = df_1["t_oa"].tolist()

        t_ra__max = np.max(t_ra)
        t_ra__75 = np.percentile(t_ra, 75)
        t_ra__mean = np.mean(t_ra)
        t_ra__std = np.std(t_ra)
        t_ra__25 = np.percentile(t_ra, 25)
        t_ra__min = np.min(t_ra)

        q_cool__max = np.max(q_cool)
        q_cool__75 = np.percentile(q_cool, 75)
        q_cool__mean = np.mean(q_cool)
        q_cool__std = np.std(q_cool)
        q_cool__25 = np.percentile(q_cool, 25)
        q_cool__min = np.min(q_cool)

        q_heat__max = np.max(q_heat)
        q_heat__75 = np.percentile(q_heat, 75)
        q_heat__mean = np.mean(q_heat)
        q_heat__std = np.std(q_heat)
        q_heat__25 = np.percentile(q_heat, 25)
        q_heat__min = np.min(q_heat)

        t_oa__max = np.max(t_oa)
        t_oa__75 = np.percentile(t_oa, 75)
        t_oa__mean = np.mean(t_oa)
        t_oa__std = np.std(t_oa)
        t_oa__25 = np.percentile(t_oa, 25)
        t_oa__min = np.min(t_oa)

        correlation1, _ = pearsonr(t_ra, q_cool)
        correlation2, _ = pearsonr(t_ra, q_heat)
        correlation3, _ = pearsonr(q_heat, q_cool)
        correlation4, _ = pearsonr(t_oa, q_cool)
        correlation5, _ = pearsonr(t_oa, q_heat)
        correlation6, _ = pearsonr(t_oa, t_ra)
        length = dataset.shape[0]

        data = {
            'threshold_q_cool': [threshold_q_cool],
            'threshold_q_heat': [threshold_q_heat],
            'length': [length],

            't_ra max': [t_ra__max],
            't_ra 75 per': [t_ra__75],
            't_ra mean': [t_ra__mean],
            't_ra std': [t_ra__std],
            't_ra 25 per': [t_ra__25],
            't_ra min': [t_ra__min],

            'q_cool max': [q_cool__max],
            'q_cool 75 per': [q_cool__75],
            'q_cool mean': [q_cool__mean],
            'q_cool std': [q_cool__std],
            'q_cool 25 per': [q_cool__25],
            'q_cool min': [q_cool__min],

            'q_heat max': [q_heat__max],
            'q_heat 75 per': [q_heat__75],
            'q_heat mean': [q_heat__mean],
            'q_heat std': [q_heat__std],
            'q_heat 25 per': [q_heat__25],
            'q_heat min': [q_heat__min],

            't_oa max': [t_oa__max],
            't_oa 75 per': [t_oa__75],
            't_oa mean': [t_oa__mean],
            't_oa std': [t_oa__std],
            't_oa 25 per': [t_oa__25],
            't_oa min': [t_oa__min],

            'corr t_ra - q_cool': [correlation1],
            'corr t_ra - q_heat': [correlation2],
            'corr q_heat - q_cool': [correlation3],
            'corr t_oa - q_cool': [correlation4],
            'corr t_oa - q_heat': [correlation5],
            'corr t_oa - t_ra': [correlation6]
        }

        df = pd.DataFrame(data)

        # Check if the file exists
        if os.path.exists(path + '/results/statistics.xlsx'):
            existing_df = pd.read_excel(path + '/results/statistics.xlsx', engine='openpyxl')
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.sort_values(by=['threshold_q_cool', 'threshold_q_heat', 'train_rate'], inplace=True)

            # Save the grouped mean data to the Excel file, overwriting the existing file
            with pd.ExcelWriter(path + '/results/statistics.xlsx', engine='openpyxl',
                                mode='w') as writer:
                combined_df.to_excel(writer, index=False, sheet_name='Sheet1')

        else:
            # Save the DataFrame to a new Excel file
            with pd.ExcelWriter(path + '/results/statistics.xlsx', engine='openpyxl',
                                mode='w') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')

    else:
        continue