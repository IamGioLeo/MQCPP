# MQCPP Solver Repository

This repository contains the code written for the Mathematical Optimization exam. The article that describes the algorithm implemented for solving the minimum quasi-clique partitioning problem can be found at https://www.sciencedirect.com/science/article/pii/S0305054825000619#da1.

## Directory Structure

1. **instances/**  
   - **Description**: Directory that contains all the graphs used in the article. In `Test.py` and `Scalability.py`, only a few of them are used. 

2. **MQCPP_ILS_solver.py**  
   - **Description**: Implements the ILS (Iterated Local Search) algorithm presented in the paper for the MQCPP. 

3. **MQCPP_gurobi_solver.py**  
   - **Description**: Contains the implementation of the mathematical model proposed in the paper for the MQCPP. 

4. **Plotting_functions.py**  
   - **Description**: Provides various functions for plotting solutions given by the solvers. 

5. **Results_functions.py**  
   - **Description**: Includes functions for displaying information about the results of the solvers execution.

6. **Scalability.py**  
   - **Description**: Scalability analysis of the algorithm, performed on 21 instances (7 graphs and 3 values of gamma).

7. **Test.py**  
   - **Description**: Quick test with small instances, performed with 12 instances (4 graphs and 3 values of gamma).

8. **Utility_functions.py**  
   - **Description**: Contains general utility functions.
## Contributors

- Daniele Maijnelli
- Giovanni Leonardi
