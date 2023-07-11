import pickle5 as pickle
import matplotlib.pyplot as plt
import support_fun
from scipy.stats import pearsonr
import argparse
parser = argparse.ArgumentParser()

# Don't change the seeds
seeds = 1

parser.add_argument("--path", help="Specify the path")
parser.add_argument("--threshold_q_cool", type=int, help="Enter the threshold for q_cool")
parser.add_argument("--threshold_q_heat", type=int, help="Enter the threshold for q_heat")

args = parser.parse_args()

path = args.path
threshold_q_cool = args.threshold_q_cool
threshold_q_heat = args.threshold_q_heat

dataset_dir = path+'/processed_data/shuffled_data/matrix_'+str(seeds)+'.pkl'      # directory containing data


with open(dataset_dir, 'rb') as handle:  # load preprocessed data
    dataset_ = (pickle.load(handle))  # [timestamp, t_oa, q_cool, q_heat, t_ra]

dataset_ = np.reshape(dataset_, (int(len(dataset_) / 48), 48, 5))

indice = support_fun.generate_indice_full(dataset_, threshold_q_cool=threshold_q_cool, threshold_q_heat=threshold_q_heat)

dataset = dataset_[indice]

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

correlation1, _ = pearsonr(df_1["t_ra"].tolist(), df_1["q_cool"].tolist())
correlation2, _ = pearsonr(df_1["t_ra"].tolist(), df_1["q_heat"].tolist())
correlation3, _ = pearsonr(df_1["q_heat"].tolist(), df_1["q_cool"].tolist())
correlation4, _ = pearsonr(df_1["t_oa"].tolist(), df_1["q_cool"].tolist())
correlation5, _ = pearsonr(df_1["t_oa"].tolist(), df_1["q_heat"].tolist())
correlation6, _ = pearsonr(df_1["t_oa"].tolist(), df_1["t_ra"].tolist())
length = dataset.shape[0]

print("threshold_q_cool",threshold_q_cool)
print("threshold_q_heat",threshold_q_heat)
print("shape dataset", length)
print("correlation t_ra - q_cool", correlation1)
print("correlation t_ra - q_heat", correlation2)
print("correlation q_heat - q_cool", correlation3)
print("correlation t_oa - q_cool", correlation4)
print("correlation t_oa - q_heat", correlation5)
print("correlation t_oa - t_ra", correlation6)