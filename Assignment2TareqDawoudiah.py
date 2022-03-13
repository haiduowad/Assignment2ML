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

################################## B(4) End

########################################### Feature Engineering (B) End ################################################