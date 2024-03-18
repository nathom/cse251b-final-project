import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import re
import seaborn as sns

def csv_to_average(csv_path):
    averaged_data = {'avg_num_moves': 0, 'avg_score': 0, 'avg_max_tile': 0, 'avg_sum_tiles': 0, 'avg_time': 0}
    cnt = 0

    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for row in reader:
            averaged_data['avg_num_moves'] += float(row[1])
            averaged_data['avg_score'] += float(row[2])
            averaged_data['avg_max_tile'] += float(row[3])
            averaged_data['avg_sum_tiles'] += float(row[4])
            averaged_data['avg_time'] += float(row[6])
            cnt += 1

    averaged_data['avg_num_moves'] /= cnt
    averaged_data['avg_score'] /= cnt
    averaged_data['avg_max_tile'] /= cnt
    averaged_data['avg_sum_tiles'] /= cnt
    averaged_data['avg_time'] /= cnt
    return averaged_data

def mcts_methods_bar_graph(name, csv_file_names):
    csv_cnt = len(csv_file_names)

    # Parse the csv file name to get the arguments.
    categories, file_order = set(), []
    regx = r'\d+'
    for (c, n, v) in csv_file_names:
        r, i, m = re.findall(regx, c)
        categories.add(int(r))
        file_order.append([int(r), int(i), int(m), csv_to_average(c)])
    file_order.sort(key=lambda x: (x[0], x[2]))
    categories = list(categories)
    categories.sort()

    cat_cnt, bar_width = len(categories), 0.2
    r1 = np.arange(cat_cnt)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    for i in range(cat_cnt):
        categories[i] = "Rollout " + str(categories[i])

    if not os.path.isdir("./data/plots"):
        os.mkdir("./data/plots")

    method_names = {0: "Max Tile", 1: "Merge Score", 2: "Sum of Tiles", 3: "Weighted Sum"}
    metrics = ['avg_num_moves', 'avg_score', 'avg_max_tile', 'avg_sum_tiles', 'avg_time']
    r, c = [r1, r2, r3, r4], ['lightblue', 'skyblue', 'paleturquoise', 'cadetblue']
    for m in metrics:
        plt.figure(figsize=(10, 7))
        for i in range(4):
            data = [file_order[j][3][m] for j in range(i, csv_cnt, 4)]
            plt.bar(r[i], data, color=c[i], width=bar_width, edgecolor='grey', label=method_names[i])

        # Hardcoded values, might not work for different plots.
        for k in range(12):
            plt.text(k / 4 - ((k % 4) * 0.05), file_order[k][3][m], str(int(file_order[k][3][m])), ha='center', va='bottom')

        plt.xlabel('Rollout Count', fontsize=16)
        plt.ylabel(m, fontsize=16)
        plt.xticks([r + bar_width for r in range(cat_cnt)], categories)
        plt.title("Rollout vs " + m, fontweight='bold', fontsize=24)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True)
        plt.savefig(f"./data/plots/{name}_{m}.png")
        plt.cla()

csv_list = [("./data/monte_carlo_branch=50_ngames=100_method=0.csv", "branches_100_games_20", 100),
            ("./data/monte_carlo_branch=50_ngames=100_method=1.csv", "branches_200_games_20", 200),
            ("./data/monte_carlo_branch=50_ngames=100_method=2.csv", "branches_300_games_20", 300),
            ("./data/monte_carlo_branch=50_ngames=100_method=3.csv", "branches_400_games_20", 400),
            ("./data/monte_carlo_branch=100_ngames=100_method=0.csv", "branches_100_games_20", 100),
            ("./data/monte_carlo_branch=100_ngames=100_method=1.csv", "branches_200_games_20", 200),
            ("./data/monte_carlo_branch=100_ngames=100_method=2.csv", "branches_300_games_20", 300),
            ("./data/monte_carlo_branch=100_ngames=100_method=3.csv", "branches_400_games_20", 400),
            ("./data/monte_carlo_branch=250_ngames=100_method=0.csv", "branches_100_games_20", 100),
            ("./data/monte_carlo_branch=250_ngames=100_method=1.csv", "branches_200_games_20", 200),
            ("./data/monte_carlo_branch=250_ngames=100_method=2.csv", "branches_300_games_20", 300),
            ("./data/monte_carlo_branch=250_ngames=100_method=3.csv", "branches_400_games_20", 400)]

mcts_methods_bar_graph("mcts", csv_list)