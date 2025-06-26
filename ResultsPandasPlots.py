import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import numpy as np

actual_csv_path = 'not_git/first_real_test_ILS.csv'
test_ILS_path = '01_final_test_ILS.csv'
test_gurobi_path = '01_final_test_gurobi.csv'
scalability_ILS_path = '01_final_scalability_ILS.csv'
scalability_gurobi_path = '01_final_scalability_gurobi.csv'


def csv_to_table(ILS_path, gurobi_path):
    df1 = pd.read_csv(ILS_path)
    df2 = pd.read_csv(gurobi_path)

    df = pd.concat([df1, df2], ignore_index=True)
    df["Graph"] = df["Graph"].str.replace("instances/", "", regex=False)
    #df["Time"] = pd.to_timedelta(df["Time"]).dt.total_seconds()
    df["Time"] = df["Time"].apply(lambda x: np.trunc(x * 1000) / 1000)
    df["Initial_solution_time"] = df["Initial_solution_time"].apply(lambda x: np.trunc(x * 1000) / 1000)
    df["Best_solution_time"] = df["Best_solution_time"].apply(lambda x: np.trunc(x * 1000) / 1000)


    df_to_print = df[["Graph", "Gamma", "Time", "Initial_solution_time","Best_solution_time","Best_solution_size"]].rename(columns={"Initial_solution_time": "IS_time","Best_solution_time": "S_time","Best_solution_size": "S"})

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    columns_to_print = ["Graph", "Gamma", "Time", "IS_time", "S_time", "S"]
    print(df_to_print[columns_to_print])

csv_to_table(test_ILS_path,test_gurobi_path)

def boxplot_log_time_for_gamma(csv_path):
    df = pd.read_csv(csv_path)
    # df["Time"] = pd.to_timedelta(df["Time"]).dt.total_seconds()
    df["Time"] = df["Time"].astype(float)
    df["Gamma"] = df["Gamma"].astype(float)

    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Gamma", y="Time", data=df)

    plt.yscale("log")
    plt.title("Boxplot del Tempo (scala log) per Gamma")
    plt.xlabel("Gamma")
    plt.ylabel("log(Time in seconds)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


#boxplot_log_time_for_gamma(actual_csv_path)


def boxplot_log_time_for_number_of_nodes(csv_path):
    df = pd.read_csv(csv_path)
    # df["Time"] = pd.to_timedelta(df["Time"]).dt.total_seconds()
    df["Time"] = df["Time"].astype(float)

    df["#nodes"] = df["#nodes"].astype(int)

    plt.figure(figsize=(8, 5))
    sns.boxplot(x="#nodes", y="Time", data=df)

    plt.yscale("log")
    plt.title("Boxplot del Tempo (scala log) per numero di nodi")
    plt.xlabel("#nodes")
    plt.ylabel("log(Time in seconds)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


#boxplot_log_time_for_number_of_nodes(actual_csv_path)


def boxplot_log_time_for_number_of_edges(csv_path):
    df = pd.read_csv(csv_path)
    # df["Time"] = pd.to_timedelta(df["Time"]).dt.total_seconds()
    df["Time"] = df["Time"].astype(float)

    df["#edges"] = df["#edges"].astype(int)

    plt.figure(figsize=(8, 5))
    sns.boxplot(x="#edges", y="Time", data=df)

    plt.yscale("log")
    plt.title("Boxplot del Tempo (scala log) per numero di nodi")
    plt.xlabel("#edges")
    plt.ylabel("log(Time in seconds)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


#boxplot_log_time_for_number_of_edges(actual_csv_path)


def time_scatterplot_for_nodes_and_gamma(csv_path):
    df = pd.read_csv(csv_path)
    # df["Time"] = pd.to_timedelta(df["Time"]).dt.total_seconds()
    df["Time"] = df["Time"].astype(float)

    df["Gamma"] = df["Gamma"].astype(float)
    df["#nodes"] = df["#nodes"].astype(int)

    # Crea scatter plot
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        x=df["Gamma"],
        y=df["#nodes"],
        c=df["Time"],
        cmap="coolwarm",
        norm=LogNorm(),
        edgecolors='face',
        s=50  # dimensione dei cerchietti
    )

    plt.yscale("log")

    cbar = plt.colorbar(sc)
    cbar.set_label("Tempo (secondi)")

    plt.xlabel("Gamma")
    plt.ylabel("Numero di nodi")
    plt.title("Scatter: Tempo in funzione di Gamma e Numero di Nodi")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#time_scatterplot_for_nodes_and_gamma(actual_csv_path)


def time_scatterplot_for_nodes_and_edges(csv_path):
    df = pd.read_csv(csv_path)
    # df["Time"] = pd.to_timedelta(df["Time"]).dt.total_seconds()
    df["Time"] = df["Time"].astype(float)

    df["#edges"] = df["#edges"].astype(int)
    df["#nodes"] = df["#nodes"].astype(int)

    # Crea scatter plot
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        x=df["#nodes"],
        y=df["#edges"],
        c=df["Time"],
        cmap="coolwarm",
        norm=LogNorm(),
        edgecolors='face',
        s=50  # dimensione dei cerchietti
    )

    plt.yscale("log")

    cbar = plt.colorbar(sc)
    cbar.set_label("Tempo (secondi)")

    plt.xlabel("Numero di nodi")
    plt.ylabel("Numero di archi")
    plt.title("Scatter: Tempo in funzione del numero di nodi e del numero di archi")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#time_scatterplot_for_nodes_and_edges(actual_csv_path)


def time_scatterplot_for_nodes_and_gamma_labels(csv_path):
    df = pd.read_csv(csv_path)
    # df["Time"] = pd.to_timedelta(df["Time"]).dt.total_seconds()
    df["Time"] = df["Time"].astype(float)
    df["Gamma"] = df["Gamma"].astype(str)
    df["#nodes"] = df["#nodes"].astype(int)

    unique_gammas = sorted(df["Gamma"].unique())
    palette = sns.color_palette("tab10", n_colors=len(unique_gammas))  # o "Set1", "Dark2", ecc.
    color_map = {gamma: palette[i] for i, gamma in enumerate(unique_gammas)}
    df["Color"] = df["Gamma"].map(color_map)

    # Crea scatter plot
    plt.figure(figsize=(8, 6))
    for gamma in unique_gammas:
        subset = df[df["Gamma"] == gamma]
        plt.scatter(
            subset["#nodes"],
            subset["Time"],
            label=f"Gamma = {gamma}",
            color=color_map[gamma],
            edgecolors='black',
            s=50
        )

    plt.yscale("log")

    plt.xlabel("Numero di nodi")
    plt.ylabel("Tempo (secondi)")
    plt.title("Scatter: Tempo in funzione di Numero di Nodi e Gamma")
    plt.legend(title="Gamma")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#time_scatterplot_for_nodes_and_gamma_labels(actual_csv_path)


def time_for_density_gamma_labels(csv_path):
    df = pd.read_csv(csv_path)
    df["Density"] = 2 * df["#edges"] / (df["#nodes"] * (df["#nodes"] - 1))

    # Conversione delle colonne necessarie
    df["Time"] = df["Time"].astype(float)

    # Crea il grafico
    plt.figure(figsize=(8, 6))
    plt.scatter(
        df["Density"],
        df["Time"],
        c=df["Gamma"],
        cmap="tab10",
        edgecolors='black',
        s=50
    )

    plt.yscale("log")
    plt.xlabel("Densità")
    plt.ylabel("Tempo (secondi)")
    plt.title("Scatter: Tempo in funzione della densità")
    plt.grid(True)

    cbar = plt.colorbar()
    cbar.set_label("Gamma")

    plt.tight_layout()
    plt.show()

#time_for_density_gamma_labels(actual_csv_path)



def jointplot_time_vs_density(csv_path):
    df = pd.read_csv(csv_path)
    df["Density"] = 2 * df["#edges"] / (df["#nodes"] * (df["#nodes"] - 1))
    df["Time"] = df["Time"].astype(float)

    sns.jointplot(
        x="Density", y="Time", data=df,
        kind="reg",  # lineare regressione
        height=8,
        marginal_kws=dict(bins=30, fill=True)
    )
    plt.suptitle("Jointplot: Time vs Density with regression", y=1.02)
    plt.show()

#jointplot_time_vs_density(actual_csv_path)

def best_solution_comparison_ILS_gurobi(ILS_path, gurobi_path):
    df1 = pd.read_csv(ILS_path)
    df2 = pd.read_csv(gurobi_path)

    df = pd.concat([df1, df2], ignore_index=True)

    df["Instance"] = df.apply(lambda row: f"{row['Graph']} {row['Gamma']}", axis=1)

    grouped = df.groupby(["Instance", "Solver"])["Best_solution_size"].mean().reset_index()

    pivot = grouped.pivot(index="Instance", columns="Solver", values="Best_solution_size")

    pivot.plot(kind="bar", figsize=(12, 6))
    plt.ylabel("Average Best Solution Size")
    plt.title("Comparison of Best Solution Size by Instance and Solver")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.legend(title="Solver")
    plt.grid(True, axis='y')
    plt.show()

best_solution_comparison_ILS_gurobi(scalability_ILS_path, scalability_gurobi_path)

def time_comparison_ILS_gurobi(ILS_path, gurobi_path):
    df1 = pd.read_csv(ILS_path)
    df2 = pd.read_csv(gurobi_path)

    df = pd.concat([df1, df2], ignore_index=True)

    df["Instance"] = df.apply(lambda row: f"{row['Graph']} {row['Gamma']}", axis=1)

    grouped = df.groupby(["Instance", "Solver"])["Best_solution_time"].mean().reset_index()

    pivot = grouped.pivot(index="Instance", columns="Solver", values="Best_solution_time")

    pivot.plot(kind="bar", figsize=(12, 6))
    plt.yscale("log")
    plt.ylabel("Average Best Solution Size")
    plt.title("Comparison of Best Solution Time by Instance and Solver")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.legend(title="Solver")
    plt.grid(True, axis= 'y')
    plt.show()

time_comparison_ILS_gurobi(scalability_ILS_path, scalability_gurobi_path)