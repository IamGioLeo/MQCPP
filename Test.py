from Function import function
import random
import numpy as np


test_instances_file_paths = ['instances/Memento.txt', 'instances/the_x_files.txt', 'instances/high-tech.txt', 'instances/mexican.paj']

seed = 40
random.seed(seed)
np.random.seed(seed)
gammas = [0.3, 0.6, 0.8]
cutoff_times = [3, 5]
cutoff_time_gurobi = 10
plot = True
gurobi_flag = True
test_csv_path = '01_final_test'

function(test_instances_file_paths, gammas, cutoff_times, plot, gurobi_flag, test_csv_path, cutoff_time_gurobi)
