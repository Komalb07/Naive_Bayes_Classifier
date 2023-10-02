# Naive_Bayes_Classifier

-> This is the implementation of Naive Bayes classifier from scratch. The data is adult census data taken from UC Irvine learning repository.
-> Firstly, the missing values were handled by replacing them with the mode value of the respective column. The data has to be discretized before proceeding for further step. So, to estimate the bin value for the process of discretization we need to visualize the data. For that, matplotlib and seaborn libraries were used. 
-> After selecting suitable bin values and disctrizated the continuous features, Prior and likelihood were calculated and mulitplied to obtain posterior probability. In this way Naive bayes classifier was implemented. 
-> K-Fold cross validation was performed to validate the model. Also Confusion matrix, accuracy score, precision and f1 score were calculated without using any built-in functions.
Finally an accuracy score of 82.53% was achieved
