import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

############################################### EDA (A) Start ##########################################################

# Importing the patent information
excelDataFrame = pd.read_csv(r'heart.csv')

# Changing Sex to int values (Male = 0, Female = 1)
excelDataFrame["SexInt"] = ""
for index, row in excelDataFrame.iterrows():
    if row["Sex"] == "F":
        excelDataFrame.at[index, 'SexInt'] = int(0)
    elif row["Sex"] == "M":
        excelDataFrame.at[index, 'SexInt'] = int(1)

# Changing ChestPainType to int values (ATA = 0, NAP = 1, ASY = 2, TA = 3)
excelDataFrame["ChestPainTypeInt"] = ""
for index, row in excelDataFrame.iterrows():
    if row["ChestPainType"] == "ATA":
        excelDataFrame.at[index, 'ChestPainTypeInt'] = int(0)
    elif row["ChestPainType"] == "NAP":
        excelDataFrame.at[index, 'ChestPainTypeInt'] = int(1)
    elif row["ChestPainType"] == "ASY":
        excelDataFrame.at[index, 'ChestPainTypeInt'] = int(2)
    elif row["ChestPainType"] == "TA":
        excelDataFrame.at[index, 'ChestPainTypeInt'] = int(3)

# Changing RestingECG to int values (Normal = 0, ST = 1, LVH = 2)
excelDataFrame["RestingECGInt"] = ""
for index, row in excelDataFrame.iterrows():
    if row["RestingECG"] == "Normal":
        excelDataFrame.at[index, 'RestingECGInt'] = int(0)
    elif row["RestingECG"] == "ST":
        excelDataFrame.at[index, 'RestingECGInt'] = int(1)
    elif row["RestingECG"] == "LVH":
        excelDataFrame.at[index, 'RestingECGInt'] = int(2)

# Changing ExerciseAngina to int values (N = 0, Y = 1)
excelDataFrame["ExerciseAnginaInt"] = ""
for index, row in excelDataFrame.iterrows():
    if row["ExerciseAngina"] == "N":
        excelDataFrame.at[index, 'ExerciseAnginaInt'] = int(0)
    elif row["ExerciseAngina"] == "Y":
        excelDataFrame.at[index, 'ExerciseAnginaInt'] = int(1)

# Changing ST_Slope to int values (Up = 0, Flat = 1, Down = 2)
excelDataFrame["ST_SlopeInt"] = ""
for index, row in excelDataFrame.iterrows():
    if row["ST_Slope"] == "Up":
        excelDataFrame.at[index, 'ST_SlopeInt'] = int(0)
    elif row["ST_Slope"] == "Flat":
        excelDataFrame.at[index, 'ST_SlopeInt'] = int(1)
    elif row["ST_Slope"] == "Down":
        excelDataFrame.at[index, 'ST_SlopeInt'] = int(2)

################################## A(1) Start

# Plotting histograms for numerical columns
excelDataFrame.hist(column='Age')
excelDataFrame.hist(column='RestingBP')
excelDataFrame.hist(column='Cholesterol')
excelDataFrame.hist(column='FastingBS')
excelDataFrame.hist(column='MaxHR')
excelDataFrame.hist(column='Oldpeak')
excelDataFrame.hist(column='HeartDisease')
plt.show()

################################## A(1) End

################################## A(2) Start

# Creating arrays containing ages for positive and negative cases
positiveCaseAge = []
negativeCaseAge = []
for index, row in excelDataFrame.iterrows():
    positiveCaseAge.append(row["Age"]) if row["HeartDisease"] == 1 else negativeCaseAge.append(row["Age"])

# Plotting histogram
sns.displot([positiveCaseAge,negativeCaseAge])
plt.suptitle('Positive & Negative Cases vs Ages')
plt.legend(loc='upper left', labels=['Positive Case', 'Negative Case'])
plt.show()

# Creating arrays containing ages for positive male and female cases
positiveCaseMaleAge = []
positiveCaseFemaleAge = []
for index, row in excelDataFrame.iterrows():
    if row["HeartDisease"] == 1 and row["Sex"] == 'M':
        positiveCaseMaleAge.append(row["Age"])
    elif row["HeartDisease"] == 1 and row["Sex"] == 'F':
        positiveCaseFemaleAge.append(row["Age"])

# Plotting box plot and printing medians
boxPlot = plt.boxplot([positiveCaseMaleAge,positiveCaseFemaleAge])
plt.suptitle('Box Plots for Positive Male/Female Cases')
plt.legend(loc='upper left',labels=['Left: Positive Male Case', 'Right: Positive Female Case'])
plt.show()
boxPlotMedians = [item.get_ydata()[0] for item in boxPlot['medians']]
print("The median age for positive male cases is "+str(boxPlotMedians[0]))
print("The median age for positive female cases is "+str(boxPlotMedians[1]))
print("The median age for positive female cases is higher than that of males")

################################## A(2) End

################################## A(3) Start

numericDataframe = excelDataFrame.filter(['Age','SexInt','ChestPainTypeInt','RestingBP','Cholesterol','FastingBS', 'RestingECGInt','MaxHR','ExerciseAnginaInt','Oldpeak','ST_SlopeInt','HeartDisease'], axis=1)
numericDataframe["Age"] = pd.to_numeric(numericDataframe["Age"])
numericDataframe["SexInt"] = pd.to_numeric(numericDataframe["SexInt"])
numericDataframe["ChestPainTypeInt"] = pd.to_numeric(numericDataframe["ChestPainTypeInt"])
numericDataframe["RestingBP"] = pd.to_numeric(numericDataframe["RestingBP"])
numericDataframe["Cholesterol"] = pd.to_numeric(numericDataframe["Cholesterol"])
numericDataframe["FastingBS"] = pd.to_numeric(numericDataframe["FastingBS"])
numericDataframe["RestingECGInt"] = pd.to_numeric(numericDataframe["RestingECGInt"])
numericDataframe["MaxHR"] = pd.to_numeric(numericDataframe["MaxHR"])
numericDataframe["ExerciseAnginaInt"] = pd.to_numeric(numericDataframe["ExerciseAnginaInt"])
numericDataframe["Oldpeak"] = pd.to_numeric(numericDataframe["Oldpeak"])
numericDataframe["ST_SlopeInt"] = pd.to_numeric(numericDataframe["ST_SlopeInt"])
numericDataframe["HeartDisease"] = pd.to_numeric(numericDataframe["HeartDisease"])

corrMatrix = numericDataframe.corr()
sns.heatmap(corrMatrix, annot=True)
plt.suptitle('Heat Map Between Predictor Variables')
plt.show()

################################## A(3) End

################################################# EDA (A) End ##########################################################

########################################## Feature Engineering (B) Start ###############################################

################################## B(1) Start

# Check for empty cells and removing them
emptyCells = np.where(pd.isnull(excelDataFrame))
for index in range(len(emptyCells[0])):
    print("Empty cell found at ["+str(emptyCells[0][index])+", "+str(emptyCells[1][index])+"]")
    print("Removing row number "+str(emptyCells[0][index]))
    excelDataFrame = excelDataFrame.drop(emptyCells[0][index])

################################## B(1) End

################################## B(2) Start

# Removing negative or 0 Cholesterol
# Removing negative Oldpeak
for index, row in excelDataFrame.iterrows():
    if row['Cholesterol'] < 1:
        print("Row "+str(index)+" has a 0 or negative Cholesterol. Removing it from our list.")
        excelDataFrame = excelDataFrame.drop(index)
    elif row['Oldpeak'] < 0:
        print("Row "+str(index)+" has a negative Oldpeak. Removing it from our list.")
        excelDataFrame = excelDataFrame.drop(index)

################################## B(2) End

################################## B(3) Start
################################## B(3) End

################################## B(4) Start
# Done from lines 12 to 58
################################## B(4) End

################################## B(4) Start

from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
standardScaler.fit(numericDataframe)
scaledNumericDataframe = standardScaler.transform(numericDataframe)
scaledNumericDataframe = pd.DataFrame(scaledNumericDataframe, columns = ['Age','SexInt','ChestPainTypeInt','RestingBP','Cholesterol','FastingBS', 'RestingECGInt','MaxHR','ExerciseAnginaInt','Oldpeak','ST_SlopeInt','HeartDisease'])

################################## B(4) End

########################################### Feature Engineering (B) End ################################################

######################################### Model Development I (C) Start ################################################

################################## C(1) Start

xDataFrame = scaledNumericDataframe.drop(['HeartDisease'], axis=1).squeeze()
yDataFrame = numericDataframe.filter(['HeartDisease'], axis=1).squeeze()

from sklearn.model_selection import train_test_split
XTrainSet, XTestSet, yTrainSet, yTestSet = train_test_split(xDataFrame,yDataFrame, stratify=yDataFrame, test_size=0.3,random_state=0)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Function that gets the cross validation score
def getCrossValScore(clfName, name):
    score = cross_val_score(clfName, XTrainSet, yTrainSet, cv=5)
    print("The 5 fold cross validation score for "+name+" is : " + str(score))
    print(name+": %0.2f accuracy with a standard deviation of %0.2f" % (score.mean(), score.std()))

# Function that gets the classification report
def getClassificationReport(predicted, name):
    print(name+": Classification report:")
    try:
        print(metrics.classification_report(yTestSet.to_numpy().tolist(), predicted.tolist()))
    except AttributeError:
        print(metrics.classification_report(yTestSet.to_numpy().tolist(), predicted))

# Getting number of correct guesses
def getCorrectGuesses(predicted, name):
    correct = 0
    total = 0
    yTestSetNew = yTestSet.tolist()
    try:
        predictedNew = predicted.tolist()
    except AttributeError:
        predictedNew = predicted
    for result in range(len(yTestSetNew)):
        total = total + 1
        if yTestSetNew[result] == predictedNew[result]:
            correct = correct + 1
    print(str(correct) + " correct predictions out of " + str(total) + " for "+name)
    print("The "+name+" percentage of the correct predictions is: " + str(correct / total))

################ KNN (k=5) Start

from sklearn.neighbors import KNeighborsClassifier
KNeighborsClf = Pipeline([('clf', KNeighborsClassifier(n_neighbors=5)), ])
KNeighborsClf.fit(XTrainSet, yTrainSet)
KNeighborsClfPredicted = KNeighborsClf.predict(XTestSet)

# Getting the cross validations score
getCrossValScore(KNeighborsClf, "KNeighborsClf")
# Getting the classification report
getClassificationReport(KNeighborsClfPredicted, "KNeighborsClf")
# Getting number of correct guesses
getCorrectGuesses(KNeighborsClfPredicted, "KNeighborsClf")
# Getting the confusion matrix
KNeighborsClfCm = confusion_matrix(yTestSet.to_numpy().tolist(), KNeighborsClfPredicted)
print("Confusion matrix for KNeighborsClf:")
print(KNeighborsClfCm)

################ KNN (k=5) End

################ SVM (kernel = rbf) Start

from sklearn import svm
SvmClf = Pipeline([('clf', svm.SVC( kernel='rbf', probability=True)), ])
SvmClf.fit(XTrainSet, yTrainSet)
SvmClfPredicted = SvmClf.predict(XTestSet)

# Getting the cross validations score
getCrossValScore(SvmClf, "SvmClf")
# Getting the classification report
getClassificationReport(SvmClfPredicted, "SvmClf")
# Getting number of correct guesses
getCorrectGuesses(SvmClfPredicted, "SvmClf")
# Getting the confusion matrix
SvmClfCm = confusion_matrix(yTestSet.to_numpy().tolist(), SvmClfPredicted)
print("Confusion matrix for SvmClf:")
print(SvmClfCm)

################ SVM (kernel = rbf) End

################ DT Start

from sklearn.tree import DecisionTreeClassifier

# Finding optimal tree
from sklearn.model_selection import GridSearchCV
DecisionTreeClassifierClfGrid = DecisionTreeClassifier()
gridParameters = {'max_depth': [1, 3, 5, 7, 9, 11, 13, 15]}
DecisionTreeClassifierClfGirdScore = GridSearchCV(DecisionTreeClassifierClfGrid, param_grid=gridParameters, cv=5)
DecisionTreeClassifierClfGirdScore.fit(XTrainSet, yTrainSet)
print("Optimal depth of the decision tree: ",DecisionTreeClassifierClfGirdScore.best_params_)

DecisionTreeClassifierClf = Pipeline([('clf', DecisionTreeClassifier(max_depth=5)), ])
DecisionTreeClassifierClf.fit(XTrainSet, yTrainSet)
DecisionTreeClassifierClfPredicted = DecisionTreeClassifierClf.predict(XTestSet)

# Getting the cross validations score
getCrossValScore(DecisionTreeClassifierClf, "DecisionTreeClassifierClf")
# Getting the classification report
getClassificationReport(DecisionTreeClassifierClfPredicted, "DecisionTreeClassifierClf")
# Getting number of correct guesses
getCorrectGuesses(DecisionTreeClassifierClfPredicted, "DecisionTreeClassifierClf")
# Getting the confusion matrix
DecisionTreeClassifierClfCm = confusion_matrix(yTestSet.to_numpy().tolist(), DecisionTreeClassifierClfPredicted)
print("Confusion matrix for DecisionTreeClassifierClf:")
print(DecisionTreeClassifierClfCm)

################ DT End

################ XGboot Start

from sklearn.ensemble import GradientBoostingClassifier
GradientBoostingClf = Pipeline([('clf', GradientBoostingClassifier()), ])
GradientBoostingClf.fit(XTrainSet, yTrainSet)
GradientBoostingClfPredicted = GradientBoostingClf.predict(XTestSet)

# Getting the cross validations score
getCrossValScore(GradientBoostingClf, "GradientBoostingClf")
# Getting the classification report
getClassificationReport(GradientBoostingClfPredicted, "GradientBoostingClf")
# Getting number of correct guesses
getCorrectGuesses(GradientBoostingClfPredicted, "GradientBoostingClf")
# Getting the confusion matrix
GradientBoostingClfCm = confusion_matrix(yTestSet.to_numpy().tolist(), GradientBoostingClfPredicted)
print("Confusion matrix for GradientBoostingClf:")
print(GradientBoostingClfCm)

################ XGboot End

from sklearn.ensemble import VotingClassifier

MajorityVotingSoftClf = VotingClassifier(estimators=[('KNN', KNeighborsClf), ('SVM', SvmClf), ('DT', DecisionTreeClassifierClf), ('XGboost', GradientBoostingClf)], voting='soft')
MajorityVotingSoftClf = MajorityVotingSoftClf.fit(XTrainSet, yTrainSet)
MajorityVotingSoftClfPredicted = MajorityVotingSoftClf.predict(XTestSet)

MajorityVotingHardClf = VotingClassifier(estimators=[('KNN', KNeighborsClf), ('SVM', SvmClf), ('DT', DecisionTreeClassifierClf), ('XGboost', GradientBoostingClf)], voting='hard')
MajorityVotingHardClf = MajorityVotingHardClf.fit(XTrainSet, yTrainSet)
MajorityVotingHardClfPredicted = MajorityVotingHardClf.predict(XTestSet)

# Getting the cross validations score
getCrossValScore(MajorityVotingSoftClf, "MajorityVotingSoftClf")
# Getting the classification report
getClassificationReport(MajorityVotingSoftClfPredicted, "MajorityVotingSoftClf")
# Getting number of correct guesses
getCorrectGuesses(MajorityVotingSoftClfPredicted, "MajorityVotingSoftClfPredicted")
# Getting the confusion matrix
MajorityVotingSoftClfCm = confusion_matrix(yTestSet.to_numpy().tolist(), MajorityVotingSoftClfPredicted)
print("Confusion matrix for MajorityVotingSoftClf:")
print(MajorityVotingSoftClfCm)

# Getting the cross validations score
getCrossValScore(MajorityVotingHardClf, "MajorityVotingHardClf")
# Getting the classification report
getClassificationReport(MajorityVotingHardClfPredicted, "MajorityVotingHardClf")
# Getting number of correct guesses
getCorrectGuesses(MajorityVotingHardClfPredicted, "MajorityVotingHardClf")
# Getting the confusion matrix
MajorityVotingHardClfCm = confusion_matrix(yTestSet.to_numpy().tolist(), MajorityVotingHardClfPredicted)
print("Confusion matrix for MajorityVotingHardClf:")
print(MajorityVotingHardClfCm)

################################## C(1) End

########################################## Model Development I (C) End #################################################

####################################### Model Development II (D) Start #################################################

################################## D(1) Start

from keras.models import Sequential
from keras.layers import Dense

# Model 1: ReLU for first 2 layers, output layer is sigmoid to get outputs between 0 and 1
kerasModel = Sequential()
kerasModel.add(Dense(12, input_dim=11, activation='relu'))
kerasModel.add(Dense(8, activation='relu'))
kerasModel.add(Dense(1, activation='sigmoid'))

# Model 2: Tanh for first 2 layers, output layer is sigmoid to get outputs between 0 and 1
kerasModel2 = Sequential()
kerasModel2.add(Dense(12, input_dim=11, activation='tanh'))
kerasModel2.add(Dense(8, activation='tanh'))
kerasModel2.add(Dense(1, activation='sigmoid'))

# Model 3: Sigmoid for first 2 layers, output layer is sigmoid to get outputs between 0 and 1
kerasModel3 = Sequential()
kerasModel3.add(Dense(12, input_dim=11, activation='sigmoid'))
kerasModel3.add(Dense(8, activation='sigmoid'))
kerasModel3.add(Dense(1, activation='sigmoid'))

kerasModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
kerasModel.fit(XTrainSet, yTrainSet, epochs=150, batch_size=10)

kerasModel2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
kerasModel2.fit(XTrainSet, yTrainSet, epochs=150, batch_size=10)

kerasModel3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
kerasModel3.fit(XTrainSet, yTrainSet, epochs=150, batch_size=10)

# Getting model accuracy
_, accuracy = kerasModel.evaluate(XTrainSet, yTrainSet)
print('\nAccuracy of kerasModelReLU: %.2f' % (accuracy*100))

# Getting classification report
kerasModelPredicted = kerasModel.predict(XTestSet)
kerasModelPredicted = [round(x[0]) for x in kerasModelPredicted]
# Getting the classification report
getClassificationReport(kerasModelPredicted, "kerasModelReLU")
# Getting number of correct guesses
getCorrectGuesses(kerasModelPredicted, "kerasModelReLU")
# Getting the confusion matrix
kerasModelCm = confusion_matrix(yTestSet.to_numpy().tolist(), kerasModelPredicted)
print("Confusion matrix for kerasModelReLU:")
print(kerasModelCm)

# Getting model accuracy
_, accuracy2 = kerasModel2.evaluate(XTrainSet, yTrainSet)
print('\nAccuracy of kerasModelTanh: %.2f' % (accuracy2*100))

# Getting classification report
kerasModel2Predicted = kerasModel2.predict(XTestSet)
kerasModel2Predicted = [round(x[0]) for x in kerasModel2Predicted]
# Getting the classification report
getClassificationReport(kerasModel2Predicted, "kerasModelTanh")
# Getting number of correct guesses
getCorrectGuesses(kerasModel2Predicted, "kerasModelTanh")
# Getting the confusion matrix
kerasModel2Cm = confusion_matrix(yTestSet.to_numpy().tolist(), kerasModel2Predicted)
print("Confusion matrix for kerasModelTanh:")
print(kerasModel2Cm)

# Getting model accuracy
_, accuracy3 = kerasModel3.evaluate(XTrainSet, yTrainSet)
print('\nAccuracy of kerasModelSigmoid: %.2f' % (accuracy3*100))

# Getting classification report
kerasModel3Predicted = kerasModel3.predict(XTestSet)
kerasModel3Predicted = [round(x[0]) for x in kerasModel3Predicted]
# Getting the classification report
getClassificationReport(kerasModel3Predicted, "kerasModelSigmoid")
# Getting number of correct guesses
getCorrectGuesses(kerasModel3Predicted, "kerasModelSigmoid")
# Getting the confusion matrix
kerasModel3Cm = confusion_matrix(yTestSet.to_numpy().tolist(), kerasModel3Predicted)
print("Confusion matrix for kerasModelSigmoid:")
print(kerasModel3Cm)

################################## D(1) End

########################################### Model Comparison (E) Start #################################################

################################## E(1) Start

# def getReportParameters(clfName):
#     total = sum(sum(clfName))
#     accuracy = (clfName[0, 0] + clfName[1, 1]) / clfName
#     percision = clfName[1, 1] / (clfName[1, 1] + clfName[0, 1])
#     recall = clfName[1, 1] / (clfName[1, 0] + clfName[1, 1])
#     f1 = 2 * (recall * percision) / (recall + percision)
#     support = clfName[0, 0] / (clfName[0, 0] + clfName[0, 1])
#     return total, accuracy, percision, recall, f1, support
#
# KNeighborsClfTotal, KNeighborsClfAccuracy, KNeighborsClfPercision, KNeighborsClfRecall, KNeighborsClfF1, KNeighborsClfSupport = getReportParameters(KNeighborsClf)
#
# print(KNeighborsClfTotal)
# print(KNeighborsClfAccuracy)
# print(KNeighborsClfPercision)
# print(KNeighborsClfRecall)
# print(KNeighborsClfF1)
# print(KNeighborsClfSupport)

################################## E(1) End

############################################# Model Comparison (E) End #################################################