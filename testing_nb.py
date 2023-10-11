import pandas as pd
from Naive_Bayes_Classifier.naive_bayes import Naive_Bayes
import numpy as np

def identify_null_attributes(data_frame):
    rows = len(data_frame)
    cols = list(data_frame.columns)

    unique_dict = {}
    for i in cols:
        unique_dict[i] = data_frame[i].unique()

    dict = {}
    for i in range(len(cols)):
        cout = 0
        for j in range(rows):
            if data_frame.iloc[j, i] == ' ?':
                cout += 1
        dict[cols[i]] = cout

    return dict

df = pd.read_csv('/Users/komalb/Downloads/adult/adult_test.csv')

null_attr = identify_null_attributes(df)

for i in df:
    if dict[i]!= 0:
        df[i].replace(' ?', pd.NA, inplace=True)

df = df.dropna()
df3 = pd.read_csv('DF3.csv')

age_bins = 15
fnlwgt_bins = 25
education_num_bins = 7
capital_gain_bins = 15
capital_loss_bins = 14
hours_per_week_bins = 7

age_labels = ['Age {}'.format(i) for i in range(1,age_bins+1)]
fnlwgt_labels = ['Fnlwgt {}'.format(i) for i in range(1,fnlwgt_bins+1)]
education_num_labels = ['Eduno {}'.format(i) for i in range(1,education_num_bins+1)]
capital_gain_labels = ['Cgain {}'.format(i) for i in range(1,capital_gain_bins+1)]
capital_loss_labels = ['Closs {}'.format(i) for i in range(1,capital_loss_bins+1)]
hours_per_week_labels = ['Hpw {}'.format(i) for i in range(1,hours_per_week_bins+1)]

#Continuous features are categorized
df['age'] = pd.cut(df['age'], age_bins, labels=age_labels)
df[' fnlwgt'] = pd.cut(df[' fnlwgt'], fnlwgt_bins, labels=fnlwgt_labels)
df[' education-num'] = pd.cut(df[' education-num'], education_num_bins, labels=education_num_labels)
df[' capital-gain'] = pd.cut(df[' capital-gain'], capital_gain_bins, labels=capital_gain_labels)
df[' capital-loss'] = pd.cut(df[' capital-loss'], capital_loss_bins, labels=capital_loss_labels)
df[' hours-per-week'] = pd.cut(df[' hours-per-week'], hours_per_week_bins, labels=hours_per_week_labels)

feats = df.columns[:-1]
class_label = df.columns[-1]

x_test, y_test = df[feats], list(df[class_label])

pred = Naive_Bayes(df3, x_test.values, feats, class_label)

correct_predictions = sum([1 for i in range(len(pred)) if pred[i]==y_test[i][:-1]])
total_predictions = len(y_test)
accuracy = correct_predictions / total_predictions

confusion_matrix = np.zeros((2, 2), dtype=int)
for true_label, predicted_label in zip(y_test, pred):
    if true_label == " >50K." and predicted_label == " >50K":
        confusion_matrix[0, 0] += 1
    elif true_label == " >50K." and predicted_label == " <=50K":
        confusion_matrix[0, 1] += 1
    elif true_label == " <=50K." and predicted_label == " >50K":
        confusion_matrix[1, 0] += 1
    elif true_label == " <=50K." and predicted_label == " <=50K":
        confusion_matrix[1, 1] += 1

precision = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])

recall = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])

f1 = 2 * (precision * recall) / (precision + recall)

print("Accuracy score : {:.2f}%".format(accuracy*100))
print("Confusion Matrix : ", confusion_matrix)
print("F1 Score : ",f1 )
print("Precision : ", precision)