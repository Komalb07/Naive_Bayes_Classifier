import pandas as pd
import numpy as np

#Identify Null Attributes in the given data frame
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

#Calculate Prior value
def calculate_prior(df, y):
    res = {}
    class_labels = df[y].unique()
    for i in class_labels:
        res[i] = round((list(df[y]).count(i))/len(df), 3)

    return res

#Calculate Likelihood values for each feature and sore it in a dictionary
def likelihood_vals(df, X, y):
    dic = {}
    res_dic = {}
    unq_vals = df[y].unique()
    for i in unq_vals:
        dic[i] = df[df[y]==i]

    for i in dic:
        res_dic[i] = {}
        for j in X:
           res_dic[i][j]  = {}
           unq = dic[i][j].unique()
           for k in unq:
               res_dic[i][j][k] = round((list(dic[i][j]).count(k))/len(dic[i][j]), 4)

    return res_dic

#Build a Naive Bayes Classifier
def Naive_Bayes(df,Xtest,feats,class_attr):
    unq_vals = df[class_attr].unique()
    posterior_prob = {}

    likehood = likelihood_vals(df, feats, class_attr)
    prior = calculate_prior(df, class_attr)
    res_posterior = []

    for k in range(len(Xtest)):
        posterior_prob = {}
        for i in unq_vals:
            var = 1
            for j in range(len(feats)):
                if Xtest[k][j] not in likehood[i][feats[j]]:
                    var*= 1/len(df[feats[j]])
                else:
                    var*= likehood[i][feats[j]][Xtest[k][j]]
            var*= prior[i]
            posterior_prob[i] = var
        res_posterior.append(max(posterior_prob, key=posterior_prob.get))

    return res_posterior



'''
                            /********      Data Pre-processing       *********/
'''

                                            #Import the dataset
original_df = pd.read_csv('/Users/komalb/Downloads/adult/adult_data.csv')
df2 = original_df.copy()

                #Identify the columns that consits of null values with their count - {column_name: count_of_null_Values}
null_attr = identify_null_attributes(df2)

                        #Replace the null values with their respective column's mode value
for i in df2:
    if dict[i]!= 0:
        df2[i].replace(' ?', df2[i].mode()[0], inplace=True)

df3 = df2.copy()

                                            #Discretization

#Bin values are selected through a process of trial and error while observing visualization plots
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
df3['age'] = pd.cut(df3['age'], age_bins, labels=age_labels)
df3[' fnlwgt'] = pd.cut(df3[' fnlwgt'], fnlwgt_bins, labels=fnlwgt_labels)
df3[' education-num'] = pd.cut(df3[' education-num'], education_num_bins, labels=education_num_labels)
df3[' capital-gain'] = pd.cut(df3[' capital-gain'], capital_gain_bins, labels=capital_gain_labels)
df3[' capital-loss'] = pd.cut(df3[' capital-loss'], capital_loss_bins, labels=capital_loss_labels)
df3[' hours-per-week'] = pd.cut(df3[' hours-per-week'], hours_per_week_bins, labels=hours_per_week_labels)

'''
                              /*** Naive Bayes Classifier and K-Fold Cross Validation ***/       
'''

features = df3.columns[:-1]
class_label = df3.columns[-1]

k = 10
accuracy_scores = []
confusion_matrices = []
f1_scores = []
precisions = []

fold_size = len(df3)//k

for i in range(k):

    start = i * fold_size
    if i < k - 1:
        end = (i + 1) * fold_size
    else:
        end = len(df3)

    validation_data = df3.iloc[start:end]
    training_data = pd.concat([df3.iloc[0:start], df3.iloc[end:len(df3)]])

    X_train, y_train = training_data[features], training_data[class_label]
    X_val, y_val = validation_data[features], validation_data[class_label]

    y_pred = Naive_Bayes(training_data, X_val.values, features, class_label)

    #Accuracy Score
    correct_predictions = (y_pred == y_val).sum()
    total_predictions = len(y_val)
    accuracy = correct_predictions / total_predictions
    accuracy_scores.append(accuracy)

    #Confusion Matrix
    confusion_matrix = np.zeros((2, 2), dtype=int)
    for true_label, predicted_label in zip(y_val, y_pred):
        if true_label == " >50K" and predicted_label == " >50K":
            confusion_matrix[0, 0] += 1  #True Positive
        elif true_label == " >50K" and predicted_label == " <=50K":
            confusion_matrix[0, 1] += 1  #False Negative
        elif true_label == " <=50K" and predicted_label == " >50K":
            confusion_matrix[1, 0] += 1  #False Positive
        elif true_label == " <=50K" and predicted_label == " <=50K":
            confusion_matrix[1, 1] += 1  #True Negative
    confusion_matrices.append(confusion_matrix)

    #Precision
    precision = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])
    precisions.append(precision)
    recall = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])

    #F1 Score
    f1 = 2 * (precision * recall) / (precision + recall)
    f1_scores.append(f1)


print("Accuracy score : {:.2f}%".format(np.mean(accuracy_scores)*100))
print("Confusion Matrix : {}".format(np.mean(confusion_matrices, axis=0)))
print("F1 Score : {:.3f} ".format(np.mean(f1_scores)))
print("Precision : {:.2f}".format(np.mean(precisions)))