from Function import function
import random
import numpy as np


test_instances_file_paths = ['instances/Memento.txt', 'instances/the_x_files.txt', 'instances/high-tech.txt', 'instances/mexican.paj']

scalability_instances_file_paths = ['instances/p-hat500-3.mtx','instances/662_bus.mtx', 'instances/brock800-3.mtx', 'instances/email-dnc-corecipient.edges', 'instances/Batman_Returns.txt', 'instances/4-FullIns_3.txt', 'instances/p-hat500-1.mtx'] # fatte

seed = 42
random.seed(seed)
np.random.seed(seed)
gammas = [0.3]
cutoff_times = [3, 3]
cutoff_time_gurobi = 10
plot = True
gurobi_flag = True
test_csv_path = '01_final_test'

function(test_instances_file_paths, gammas, cutoff_times, plot, gurobi_flag, test_csv_path, cutoff_time_gurobi)
