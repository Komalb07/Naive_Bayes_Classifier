import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
df = pd.read_csv('nonull_adult_data.csv')

#Display continuous columns (Histogram plots)
fig,axes = plt.subplots(3,2)
hist1 = sns.histplot(df['age'],ax=axes[0,0],bins=15,edgecolor='black', kde=True, color='orange')
hist2 = sns.histplot(df[' fnlwgt'],ax=axes[0,1],bins=25,edgecolor='black', kde=True, color='b')
hist3 = sns.histplot(df[' education-num'], ax=axes[1,0], bins=7, edgecolor='black', kde=True, color='r')
hist4 = sns.histplot(df[df[' capital-gain']!=0][' capital-gain'], ax=axes[1,1], bins=10, edgecolor='black', kde=True, color='g')
hist5 = sns.histplot(df[df[' capital-loss']!=0][' capital-loss'], ax=axes[2,0], bins=14, edgecolor='black', kde=True, color='y')
hist6 = sns.histplot(df[' hours-per-week'], ax=axes[2,1], bins=7, edgecolor='black', kde=True, color='olive')

plt.show()

                            #Display remaining columns (Horizaontal Bar charts)

#Divide the remaining categories into two halves labels_1 and labels_2
labels_1 = [' workclass', ' marital-status', ' relationship',' race']
labels_with_freq = {}
num_labels_to_display = len(labels_1)

#Count unique elements and their count from respective columns
for i in labels_1:
    labels_with_freq[i] = {}
    for j in list(df[i].unique()):
        labels_with_freq[i][j] = list(df[i]).count(j)

colors = ['#ff4d4d', '#80d4ff', '#ffc966', '#999900']
fig1, axes1 = plt.subplots(2,2, figsize=(18, 14))
axes1 = axes1.flatten()

#Plot the horizontal bar charts for each category
for i,(cats, vals) in enumerate(labels_with_freq.items()):
    ax = axes1[i]
    bars = ax.barh(list(vals.keys()),list(vals.values()), color = colors[i])
    ax.set_title(cats)

    for bar, val in zip(bars, list(vals.values())):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, str(val), va='center', ha='left', fontsize=10)

plt.tight_layout()
plt.show()

#Count unique elements and their count from respective columns
labels_2 = [' native-country', ' sex', ' occupation', ' education']
labels_with_freq_2 = {}
num_labels_to_display_2 = len(labels_2)
for i in labels_2:
    labels_with_freq_2[i] = {}
    for j in list(df[i].unique()):
        labels_with_freq_2[i][j] = list(df[i]).count(j)

#Modifying ' native-country' column for the sake of visualization as there are too many unique elements in the " native-country",
#It looks crowded if we include all the elements, so we filter out those keys whose value is less than 100 and add their value to
#a new key called "Other"
for i in labels_with_freq_2:
    if i == ' native-country':
        labels_with_freq_2[i]['Other'] = 0
        mark_for_deletion = []
        for j,k in labels_with_freq_2[i].items():
            if k<100:
                labels_with_freq_2[i]['Other'] += k
                mark_for_deletion.append(j)

for i in mark_for_deletion:
    del labels_with_freq_2[' native-country'][i]

#Plot the horizontal bar charts for each category and pie chart for " sex" category as there are only two unique elements
colors = ['#ff4d4d', '#80d4ff', '#ffc966', '#999900']
fig2, axes2 = plt.subplots(2,2, figsize=(18, 14))
axes2 = axes2.flatten()

for i,(cats, vals) in enumerate(labels_with_freq_2.items()):
    if i==1:
        ax = axes2[i]
        ax.pie(vals.values() , labels=(vals.keys()), autopct='%1.1f%%')
        ax.set_title(cats)

    else:
        ax = axes2[i]
        bars_2 = ax.barh(list(vals.keys()),list(vals.values()), color = colors[i])
        ax.set_title(cats)

        for bar, val in zip(bars_2, list(vals.values())):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, str(val), va='center', ha='left', fontsize=10)

plt.tight_layout()
plt.show()