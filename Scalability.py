import random

import numpy as np

from Utility_functions import function

scalability_instances_file_paths = ['instances/Batman_Returns.txt', 'instances/4-FullIns_3.txt',
                                    'instances/p-hat500-1.mtx', 'instances/p-hat500-3.mtx', 'instances/662_bus.mtx',
                                    'instances/brock800-3.mtx', 'instances/email-dnc-corecipient.edges']

seed = 40
random.seed(seed)
np.random.seed(seed)
gammas = [0.3, 0.7, 0.9]
cutoff_times = [600, 1200]
cutoff_time_gurobi = 2400
plot = True
gurobi_flag = True
scalability_csv_path = '01_final_scalability'

function(scalability_instances_file_paths, gammas, cutoff_times, plot, gurobi_flag, scalability_csv_path,
         cutoff_time_gurobi)
