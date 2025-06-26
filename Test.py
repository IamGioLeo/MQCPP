from Function import function
import random
import numpy as np

#istanze problematiche: anna g 0.999; 'instances/494_bus.mtx'
#instances_file_paths = ['instances/Memento.txt', 'instances/the_x_files.txt', 'instances/Alien_3.txt', 'instances/high-tech.txt', 'instances/karate.mtx', 'instances/mexican.paj', 'instances/Sawmill.net', 'instances/chesapeake.mtx', 'instances/Batman_Returns.txt', 'instances/Attiro.paj', 'instances/soc-dolphins.mtx', 'instances/SanJuanSur.paj', 'instances/jean.txt', 'instances/3-FullIns_3.txt', 'instances/david.txt', 'instances/myciel6.txt', 'instances/4-FullIns_3.txt', 'instances/anna.txt']
                        # 'instances/494_bus.mtx'
#instances_file_paths = ['instances/p-hat500-1.mtx', 'instances/p-hat500-2.mtx', 'instances/p-hat500-3.mtx', 'instances/662_bus.mtx', 'instances/keller5.mtx', 'instances/brock800-3.mtx', 'instances/email-dnc-corecipient.edges', 'instances/san1000.mtx', 'instances/p-hat1000-1.mtx', 'instances/p-hat1000-2.mtx', 'instances/p-hat1000-3.mtx', 'instances/email.edges', 'instances/polblogs.mtx', 'instances/p-hat1500-3.mtx', 'instances/p-hat1500-1.mtx', 'instances/p-hat1500-2.mtx', 'instances/C2000-5.mtx', 'instances/bcsstk13.mtx', 'instances/soc-hamsterster.edges', 'instances/data.mtx']
#instances_file_paths = ['instances/Memento.txt', 'instances/the_x_files.txt', 'instances/Alien_3.txt']
#instances_file_paths = ['instances/4-FullIns_3.txt', 'instances/Batman_Returns.txt','instances/brock800-3.mtx', 'instances/data.mtx', 'instances/494_bus.mtx', 'instances/bcsstk13.mtx']

#fatte
test_instances_file_paths = ['instances/Memento.txt', 'instances/the_x_files.txt', 'instances/high-tech.txt', 'instances/mexican.paj']

scalability_instances_file_paths = ['instances/p-hat500-3.mtx','instances/662_bus.mtx', 'instances/brock800-3.mtx', 'instances/email-dnc-corecipient.edges', 'instances/Batman_Returns.txt', 'instances/4-FullIns_3.txt', 'instances/p-hat500-1.mtx'] # fatte

seed = 42
random.seed(seed)
np.random.seed(seed)
gammas = [0.3, 0.6, 0.8, 0.9]
cutoff_times = [300, 500]
plot = False
gurobi_flag = False
test_csv_path = '01_final_test'

function(test_instances_file_paths, gammas, cutoff_times, plot, gurobi_flag, test_csv_path)
