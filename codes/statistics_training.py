import pickle5 as pickle
import matplotlib.pyplot as plt
import support_fun
from scipy.stats import pearsonr
import argparse
import os
import numpy as np
import pandas as pd
parser = argparse.ArgumentParser()


parser.add_argument("--path", help="Specify the path")

args = parser.parse_args()

path = args.path

for train_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for threshold_q_cool in [0, 10, 20, 30, 40, 50]:
        for threshold_q_heat in [0, 10, 20, 30, 40, 50]:

            t_ra__max = []
            t_ra__75 = []
            t_ra__mean = []
            t_ra__std = []
            t_ra__25 = []
            t_ra__min = []

            q_cool__max = []
            q_cool__75 = []
            q_cool__mean = []
            q_cool__std = []
            q_cool__25 = []
            q_cool__min = []

            q_heat__max = []
            q_heat__75 = []
            q_heat__mean = []
            q_heat__std = []
            q_heat__25 = []
            q_heat__min = []

            t_oa__max = []
            t_oa__75 = []
            t_oa__mean = []
            t_oa__std = []
            t_oa__25 = []
            t_oa__min = []

            correlation1 = []
            correlation2 = []
            correlation3 = []
            correlation4 = []
            correlation5 = []
            correlation6 = []

            length = []

            flag = 0

            for seeds in [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 123456789, 12345678910]:

                dataset_dir = path+'/processed_data/shuffled_data/matrix_'+str(seeds)+'.pkl'      # directory containing data

                with open(dataset_dir, 'rb') as handle:  # load preprocessed data
                    dataset_ = (pickle.load(handle))  # [timestamp, t_oa, q_cool, q_heat, t_ra]

                dataset_ = np.reshape(dataset_, (int(len(dataset_) / 48), 48, 5))

                indice = support_fun.generate_indice_full(dataset_, threshold_q_cool=threshold_q_cool, threshold_q_heat=threshold_q_heat)

                dataset = dataset_[indice]

                if dataset.shape[0] < 10:
                    flag = 1
                    continue

                trX_dataset, teX_dataset, ttX_dataset = np.split(dataset, [int(train_rate * len(dataset)),int((train_rate + 0.1) * len(dataset))])

                timestamp_1 = trX_dataset[:,:,0:1]
                ttX_p_1 = trX_dataset[:,:,1:]

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


                t_ra__max = np.append(t_ra__max, np.max(t_ra))
                t_ra__75 = np.append(t_ra__75, np.percentile(t_ra, 75))
                t_ra__mean = np.append(t_ra__mean, np.mean(t_ra))
                t_ra__std = np.append(t_ra__std, np.std(t_ra))
                t_ra__25 = np.append(t_ra__25, np.percentile(t_ra, 25))
                t_ra__min = np.append(t_ra__min, np.min(t_ra))

                q_cool__max = np.append(q_cool__max, np.max(q_cool))
                q_cool__75 = np.append(q_cool__75, np.percentile(q_cool, 75))
                q_cool__mean = np.append(q_cool__mean, np.mean(q_cool))
                q_cool__std = np.append(q_cool__std, np.std(q_cool))
                q_cool__25 = np.append(q_cool__25, np.percentile(q_cool, 25))
                q_cool__min = np.append(q_cool__min, np.min(q_cool))

                q_heat__max = np.append(q_heat__max, np.max(q_heat))
                q_heat__75 = np.append(q_heat__75, np.percentile(q_heat, 75))
                q_heat__mean = np.append(q_heat__mean, np.mean(q_heat))
                q_heat__std = np.append(q_heat__std, np.std(q_heat))
                q_heat__25 = np.append(q_heat__25, np.percentile(q_heat, 25))
                q_heat__min = np.append(q_heat__min, np.min(q_heat))

                t_oa__max = np.append(t_oa__max, np.max(t_oa))
                t_oa__75 = np.append(t_oa__75, np.percentile(t_oa, 75))
                t_oa__mean = np.append(t_oa__mean, np.mean(t_oa))
                t_oa__std = np.append(t_oa__std, np.std(t_oa))
                t_oa__25 = np.append(t_oa__25, np.percentile(t_oa, 25))
                t_oa__min = np.append(t_oa__min, np.min(t_oa))

                correlation1_, _ = pearsonr(t_ra, q_cool)
                correlation2_, _ = pearsonr(t_ra, q_heat)
                correlation3_, _ = pearsonr(q_heat, q_cool)
                correlation4_, _ = pearsonr(t_oa, q_cool)
                correlation5_, _ = pearsonr(t_oa, q_heat)
                correlation6_, _ = pearsonr(t_oa, t_ra)

                correlation1 = np.append(correlation1, correlation1_)
                correlation2 = np.append(correlation2, correlation2_)
                correlation3 = np.append(correlation3, correlation3_)
                correlation4 = np.append(correlation4, correlation4_)
                correlation5 = np.append(correlation5, correlation5_)
                correlation6 = np.append(correlation6, correlation6_)

                length = np.append(length, trX_dataset.shape[0])

            if flag == 0:
                data = {
                    'threshold_q_cool': [threshold_q_cool],
                    'threshold_q_heat': [threshold_q_heat],
                    'train_rate': [train_rate],
                    'length': [np.mean(length)],

                    't_ra max': [np.mean(t_ra__max)],
                    't_ra 75 per': [np.mean(t_ra__75)],
                    't_ra mean': [np.mean(t_ra__mean)],
                    't_ra std': [np.mean(t_ra__std)],
                    't_ra 25 per': [np.mean(t_ra__25)],
                    't_ra min': [np.mean(t_ra__min)],

                    'q_cool max': [np.mean(q_cool__max)],
                    'q_cool 75 per': [np.mean(q_cool__75)],
                    'q_cool mean': [np.mean(q_cool__mean)],
                    'q_cool std': [np.mean(q_cool__std)],
                    'q_cool 25 per': [np.mean(q_cool__25)],
                    'q_cool min': [np.mean(q_cool__min)],

                    'q_heat max': [np.mean(q_heat__max)],
                    'q_heat 75 per': [np.mean(q_heat__75)],
                    'q_heat mean': [np.mean(q_heat__mean)],
                    'q_heat std': [np.mean(q_heat__std)],
                    'q_heat 25 per': [np.mean(q_heat__25)],
                    'q_heat min': [np.mean(q_heat__min)],

                    't_oa max': [np.mean(t_oa__max)],
                    't_oa 75 per': [np.mean(t_oa__75)],
                    't_oa mean': [np.mean(t_oa__mean)],
                    't_oa std': [np.mean(t_oa__std)],
                    't_oa 25 per': [np.mean(t_oa__25)],
                    't_oa min': [np.mean(t_oa__min)],

                    'corr t_ra - q_cool': [np.mean(correlation1)],
                    'corr t_ra - q_heat': [np.mean(correlation2)],
                    'corr q_heat - q_cool': [np.mean(correlation3)],
                    'corr t_oa - q_cool': [np.mean(correlation4)],
                    'corr t_oa - q_heat': [np.mean(correlation5)],
                    'corr t_oa - t_ra': [np.mean(correlation6)]
                }

                df = pd.DataFrame(data)

                # Check if the file exists
                if os.path.exists(path + '/results/statistics_training.xlsx'):
                      existing_df = pd.read_excel(path + '/results/statistics_training.xlsx', engine='openpyxl')
                      combined_df = pd.concat([existing_df, df], ignore_index=True)
                      combined_df.sort_values(by=['threshold_q_cool', 'threshold_q_heat', 'train_rate'], inplace=True)

                      # Save the grouped mean data to the Excel file, overwriting the existing file
                      with pd.ExcelWriter(path + '/results/statistics_training.xlsx', engine='openpyxl', mode='w') as writer:
                        combined_df.to_excel(writer, index=False, sheet_name='Sheet1')

                else:
                      # Save the DataFrame to a new Excel file
                      with pd.ExcelWriter(path + '/results/statistics_training.xlsx', engine='openpyxl', mode='w') as writer:
                        df.to_excel(writer, index=False, sheet_name='Sheet1')

            else:
                continue
