import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

file_path = "save/_test/log.txt"

with open(file_path, "r") as fb:
    count = 0
    flag = 0
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    for line in fb:
        line = line.strip()
        if (line == "spleen: 5 imgs meta best dice, test on 125 imgs:"
                and count <= 0):
            flag = 1
            count = 30
        elif line == "spleen: pre-train, test on 125 imgs:" and count <= 0:
            flag = 2
            count = 30
        # elif line == "OSAM trained on CT 5 imgs/task together(65 imgs), tested on liver, best loss:" and count <= 0:
        #     flag = 3
        #     count = 30
        # elif line == 'OSAM trained on CT 5 imgs/task together(65 imgs), tested on liver, best dice:' and count <= 0:
        #     flag = 4
        #     count = 30
        if count > 0:
            if (flag == 1 and line != "spleen: 5 imgs meta best dice, test on 125 imgs:"
            ):
                list1.append(float(line.split(",")[1].split(":")[-1]))
                count -= 1
            elif flag == 2 and line != "spleen: pre-train, test on 125 imgs:":
                list2.append(float(line.split(",")[1].split(":")[-1]))
                count -= 1
            # elif flag == 3 and line != "OSAM trained on CT 5 imgs/task together(65 imgs), tested on liver, best loss:":
            #     list3.append(float(line.split(",")[2].split(":")[-1]))
            #     count -= 1
            # elif flag == 4 and line != "OSAM trained on CT 5 imgs/task together(65 imgs), tested on liver, best dice:":
            #     list4.append(float(line.split(",")[2].split(":")[-1]))
            #     count -= 1

    x = list(range(len(list1) + 1))[1:]
    df = pd.DataFrame({"x": x, "list1": list1, "list2": list2})
    sns.set_style("whitegrid")
    plt.grid(color="white", linewidth=1)
    # sns.set(style="darkgrid", palette="pastel")
    sns.lineplot(x="x", y="list1", data=df, label="SSM-SAM 5 imgs", color='#0000ff', errorbar='sd', linestyle='--',
                 marker='s')
    sns.lineplot(x="x", y="list2", data=df, label="SS-SAM 5 imgs", color='#00aaff', errorbar='sd', linestyle='-',
                 marker='o')
    # sns.lineplot(x="x", y="list3", data=df, label="Adapter-SAM with pre-training", color='#ff0000', errorbar='sd', linestyle='-.',
    #              marker='v')
    # sns.lineplot(x="x", y="list4", data=df, label="Adapter-SAM with pre-training best dice", color='#00aaff', errorbar='sd',
    #              linestyle='-.',
    #              marker='^')
    plt.gca().set_facecolor("#eaeaf3")
    ax = plt.gca()
    ax.fill_between(x, np.array(list1) - 0.003, np.array(list1) + 0.003, color='blue', alpha=0.2)
    ax.set_title("DSC - spleen", fontsize=17)
    ax.set_ylabel('Dice', fontsize=15)
    ax.set_xlabel("Epoch", fontsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    plt.savefig("result_figs/DSC_5_meta_vs_original_on_spleen.pdf", format="pdf")
    plt.show()
