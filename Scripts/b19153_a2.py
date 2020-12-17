'''
Aryan Garg
B19153
+91-8219383122

Lab assignment - 2
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mode, median
from scipy import stats as s

data = pd.read_csv("pima_indians_diabetes_miss.csv")
origData = pd.read_csv("pima_indians_diabetes_original.csv")

attrs = list(data.columns) #attribute's list

# Original file descriptive analysis (for 4a and 4b)
Oattrs = list(origData.columns)
Omean = np.mean(origData, axis=0)
Omode1 = []
Omedian = []

for e in Oattrs:
    Omode1.append(s.mode(origData[e]))
    Omedian.append(median(origData[e]))
Omode = []

for e in Omode1:
    Omode.append(e[0][0])

    
# Each question has a dedicated function
def Q1():
    empCntVec = data.isnull().sum() #Count of missing values in each column
    #print(sum(empCntVec))
    plt.title("Q1) Count of NaN values in each attribute")
    plt.xlabel("Attributes")
    plt.ylabel("Empty Values")

    plt.grid(True)
    plt.bar(attrs,empCntVec)
    plt.show()

def Q2():
    oldShape = data.shape
    data.dropna(axis = 0, thresh = 7, inplace = True)
    newShape = data.shape

    print("Q2(a)")
    print("Number of records dropped with more than 2 attribute values missing:",oldShape[0]-newShape[0])
    print()

    arr = np.arange(768)
    dindices = data.index
    deletedIndices = []

    for e in arr:
        if e not in dindices:
            deletedIndices.append(e)

    print("The following records(index) were dropped with more than 2 missing attributes:")
    print(deletedIndices)
    print()

    oldShape = data.shape
    data.dropna(axis = 0, subset=['class'], inplace = True)
    newShape = data.shape

    dindices2 = data.index
    print("Q2 (b)")
    print("Number of records dropped with missing values in class attribute:",oldShape[0]-newShape[0])
    print()
    deletedIndices2 = []

    for e in arr:
        if e not in dindices2:
            deletedIndices2.append(e)

    missingClass = []
    for e in deletedIndices2:
        if e not in deletedIndices:
            missingClass.append(e)
        
    print("The following records(index) were dropped with missing class values:")
    print(missingClass)
    print()

def Q3():
    emptyAttrs = data.isnull().sum()
    totMissing = 0
    print()
    print("Missing values in each attribute:")
    for i in range(len(emptyAttrs)):
        print(attrs[i],":",emptyAttrs[i])
        totMissing += emptyAttrs[i]
    print()
    print("Total missing values in the file:",totMissing)
    return emptyAttrs
    
def Q4a(emptyAttrs):
    col_mean1 = np.nanmean(data, axis=0)
    missingValues = data.isnull()
    
    meanDict = {}

    print("Q4 (a)")
    print("Filling NaN values with attribute NaN mean...")
    for i in range(len(attrs)):
        meanDict[attrs[i]] = col_mean1[i]

    data4a = data.fillna(meanDict)
    print("Done.")
    print()
    
    # For imputed data
    col_mean = np.mean(data4a, axis = 0)
    col_mode1 = []
    col_median = []

    col_stddev = []
    for e in attrs:
        col_mode1.append(s.mode(data4a[e]))
        col_median.append(median(data4a[e]))
        col_stddev.append(np.std(data4a[e]))

    print("Stddev:",col_stddev)
    # Due to repeating values found by statistics.mode... We take the first value in the vector
    col_mode = []
    for e in col_mode1:
        col_mode.append(e[0][0])
   
    print("(a)-(i)")
    print("Comparing mean, median and mode of the two files...")
    print("  ---------------------------------------------")
    print("  Attribute   |   Mean   |   Original File Mean")
    print("  ---------------------------------------------")
    for i in range(len(attrs)):
        print("  {0:11} | {1:8.3f} | {2:8.3f}".format(attrs[i],col_mean[i],Omean[i]))

    print()
    print("  ---------------------------------------------")
    print("  Attribute   |  Median  |Original File Median")
    print("  ---------------------------------------------")
    for i in range(len(attrs)):
        print("  {0:11} | {1:8.3f} | {2:8.3f}".format(attrs[i],col_median[i],Omedian[i]))
    print()
    print("  ---------------------------------------------")
    print("  Attribute   |   Mode   |   Original File Mode")
    print("  ---------------------------------------------")
    for i in range(len(attrs)):
        print("  {0:11} | {1:8} | {2:8}".format(attrs[i],col_mode[i],Omode[i]))
    print()

    print("(a)-(ii)")
    rmse = 0
    
    missingValues.reset_index(drop = True, inplace = True)
    data4a.reset_index(drop=True, inplace = True)
    rmse = []
    for i in range(len(attrs)):
        rmse.append(0)
        if(emptyAttrs[i] == 0):
            continue
        
        for row in range(708):
            if missingValues[attrs[i]][row]:
                rmse[i] += ( data4a[attrs[i]][row] - origData[attrs[i]][row] )**2
        rmse[i] /= emptyAttrs[i]
        rmse[i] = (rmse[i]**(0.5))

    print("RMSE values (attribute-wise):")
    for i in range(len(attrs)):
        print("{0:8} -> {1:8.3f}".format(attrs[i], rmse[i]))

    plt.title("RMSE values, attribute-wise")
    plt.ylabel("RMSE")
    plt.xlabel("Attributes")
    plt.bar(attrs, rmse, color='r')
    plt.grid(True)
    plt.show()
    
def Q4b():
    missingValues = data.isnull() # Truth table of missing values in starting file: data
    
    print()
    print("Q4(b)")
    print("Replacing values by linear interpolation(column-wise)...")
    data.interpolate(method='linear', axis=0, inplace=True)
    print("[+] Done.")
    print()
    
    #For comparision with original file
    #Just like Q4a, but on starting file: data
    
    col_mean = np.mean(data, axis = 0)
    col_mode1 = []
    col_median = []

    col_stddev = []
    for e in attrs:
        col_mode1.append(s.mode(data[e]))
        col_median.append(median(data[e]))
        col_stdev.append(s.stdev(data[e]))

    col_mode = []
    for e in col_mode1:
        col_mode.append(e[0][0])

    print("Stddev:",col_stdev)
    
    print("(b)-(i)")
    print("File1 -> Linearly interpolated imputation")
    print("File2 -> Original")
    print("Comparing mean, median and mode of the two files...")
    print("  ---------------------------------------------")
    print("  Attribute   |   Mean   |   Original File Mean")
    print("  ---------------------------------------------")
    for i in range(len(attrs)):
        print("  {0:11} | {1:8.3f} | {2:8.3f}".format(attrs[i],col_mean[i],Omean[i]))

    print()
    print("  ---------------------------------------------")
    print("  Attribute   |  Median  |Original File Median")
    print("  ---------------------------------------------")
    for i in range(len(attrs)):
        print("  {0:11} | {1:8.3f} | {2:8.3f}".format(attrs[i],col_median[i],Omedian[i]))
    print()
    print("  ---------------------------------------------")
    print("  Attribute   |   Mode   |   Original File Mode")
    print("  ---------------------------------------------")
    for i in range(len(attrs)):
        print("  {0:11} | {1:8} | {2:8}".format(attrs[i],col_mode[i],Omode[i]))
    print()

    print("(b)-(ii)")
    rmse = 0
    
    missingValues.reset_index(drop = True, inplace = True)
    data.reset_index(drop=True, inplace = True)
    rmse = []
    for i in range(len(attrs)):
        rmse.append(0)
        if(emptyAttrs[i] == 0):
            continue
        
        for row in range(708):
            if missingValues[attrs[i]][row]:
                rmse[i] += ( data[attrs[i]][row] - origData[attrs[i]][row] )**2
        rmse[i] /= emptyAttrs[i]
        rmse[i] = (rmse[i]**(0.5))

    print("RMSE values (attribute-wise):")
    for i in range(len(attrs)):
        print("{0:8} -> {1:8.3f}".format(attrs[i], rmse[i]))

    plt.title("RMSE values, attribute-wise")
    plt.ylabel("RMSE")
    plt.xlabel("Attributes")
    plt.bar(attrs, rmse, color='r')
    plt.grid(True)
    plt.show()

def boxplot(title): # Boxplots for BMI and Age attributes (Q5)
    plt.subplot(121)
    plt.title("Age - boxplot")
    plt.boxplot(data['Age'])
    plt.grid(True)

    plt.subplot(122)
    plt.title("BMI - boxplot")
    plt.boxplot(data['BMI'])
    plt.grid(True)

    plt.suptitle(title)
    plt.show()

def detect_outliers(dataseries):
    # Returns list of outlier values (Q5)
    sorted(dataseries)
    q1, q3 = np.percentile(dataseries, [25,75])
    iqr = q3 - q1
    print("q1:",q1,"| q3:",q3)
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    outliers = []
    for e in dataseries:
        if (e < lower_bound) or (e > upper_bound):
            outliers.append(e)

    return outliers
    
def Q5():
    # (i) Detecting outliers and
    # Boxplots of Age and BMI parameters for outlier detection
    
    age_outliers = detect_outliers(data['Age'])
    bmi_outliers = detect_outliers(data['BMI'])

    boxplot("Before replacing outliers")
    #(ii) Replace outliers with median of attribute
    # and then plot again
    age_median = median(data['Age'])
    bmi_median = median(data['BMI'])
    
    for i in range(len(data['Age'])):
        if data['Age'][i] in age_outliers:
            data['Age'][i] = age_median

    for i in range(len(data['BMI'])):
        if data['BMI'][i] in bmi_outliers:
            data['BMI'][i] = bmi_median

    boxplot("After replacing outliers with median")
    age_outliers1 = detect_outliers(data['Age'])
    bmi_outliers1 = detect_outliers(data['BMI'])

print("-------------------- DS3 Assignment - 2 --------------------")
print()
print("Q1 Plotting Bar Chart of missing values in each attribute....")
Q1()
print()
Q2()
print()
print("Q3 Counting missing values in each attribute and the whole file...")
emptyAttrs = Q3()
print()
Q4a(emptyAttrs)
Q4b()
print()
print("Q5 Outlier detection and manipulation under progress...")
Q5()
print("[+] Program finished successfully.")
print("---------------------------- xxx ----------------------------")
