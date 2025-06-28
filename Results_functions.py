import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

test_ILS_path = '01_final_test_ILS.csv'
test_gurobi_path = '01_final_test_gurobi.csv'
scalability_ILS_path = '01_final_scalability_ILS.csv'
scalability_gurobi_path = '01_final_scalability_gurobi.csv'


def csv_to_table(ILS_path, gurobi_path):
    df1 = pd.read_csv(ILS_path)
    df2 = pd.read_csv(gurobi_path)

    df = pd.concat([df1, df2], ignore_index=True)
    df["Graph"] = df["Graph"].str.replace("instances/", "", regex=False)
    df["Time"] = df["Time"].apply(lambda x: np.trunc(x * 1000) / 1000)
    df["Initial_solution_time"] = df["Initial_solution_time"].apply(lambda x: np.trunc(x * 1000) / 1000)
    df["Best_solution_time"] = df["Best_solution_time"].apply(lambda x: np.trunc(x * 1000) / 1000)

    df_to_print = df[["Graph", "Gamma", "Time", "Initial_solution_time", "Best_solution_time", "Best_solution_size",
                      "Solver"]].rename(
        columns={"Initial_solution_time": "IS_time", "Best_solution_time": "S_time", "Best_solution_size": "S"})

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    columns_to_print = ["Graph", "Gamma", "Time", "IS_time", "S_time", "S", "Solver"]
    print(df_to_print[columns_to_print])


csv_to_table(test_ILS_path, test_gurobi_path)


def best_solution_comparison_ILS_gurobi(ILS_path, gurobi_path):
    df1 = pd.read_csv(ILS_path)
    df2 = pd.read_csv(gurobi_path)

    df = pd.concat([df1, df2], ignore_index=True)

    df["Instance"] = df.apply(lambda row: f"{row['Graph']} {row['Gamma']}", axis=1)

    grouped = df.groupby(["Instance", "Solver"])["Best_solution_size"].mean().reset_index()

    pivot = grouped.pivot(index="Instance", columns="Solver", values="Best_solution_size")

    pivot.plot(kind="bar", figsize=(12, 6))
    plt.ylabel("Best Solution Size")
    plt.title("Comparison of Best Solution Size by Instance and Solver")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.legend(title="Solver")
    plt.grid(True, axis='y')
    plt.show()


best_solution_comparison_ILS_gurobi(test_ILS_path, test_gurobi_path)


def time_comparison_ILS_gurobi(ILS_path, gurobi_path):
    df1 = pd.read_csv(ILS_path)
    df2 = pd.read_csv(gurobi_path)

    df = pd.concat([df1, df2], ignore_index=True)

    df["Instance"] = df.apply(lambda row: f"{row['Graph']} {row['Gamma']}", axis=1)

    grouped = df.groupby(["Instance", "Solver"])["Best_solution_time"].mean().reset_index()

    pivot = grouped.pivot(index="Instance", columns="Solver", values="Best_solution_time")

    pivot.plot(kind="bar", figsize=(12, 6))
    plt.yscale("log")
    plt.ylabel("Best Solution Time")
    plt.title("Comparison of Best Solution Time by Instance and Solver")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.legend(title="Solver")
    plt.grid(True, axis='y')
    plt.show()


time_comparison_ILS_gurobi(test_ILS_path, test_gurobi_path)
