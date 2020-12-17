'''
ARYAN GARG
B19153
+91-8219383122

NOTE:
Please close the created graph to obtain the next one.
The program creates one graph at a time to be more efficient.

Written in IDLE - 3.7 32-bit.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Read the dataset and parse the dates
data = pd.read_csv("landslide_data3.csv", parse_dates=['dates'])

#print(data.head())
#print(data['dates'])

cols = []
for e in data.columns:
    cols.append(e)

def Q1():
    print("A1. ***  Attributes of numerical data  ***")
    print()
    print("----------Mean----------") 
    for i in range(2,len(cols)):
        print(cols[i],":",data[cols[i]].mean())
    print()

    print("----------Median----------")
    for i in range(2,len(cols)):
        print(cols[i],":",data[cols[i]].median())
    print()

    print("----------Mode----------")
    for i in range(2,len(cols)):
        print(cols[i],":",data[cols[i]].mode())
    print()

    print("----------Min value----------")
    for i in range(2,len(cols)):
        print(cols[i],":",min(data[cols[i]]))
    print()
    
    print("----------Max value----------")
    for i in range(2,len(cols)):
        print(cols[i],":",max(data[cols[i]]))
    print()

    print("----------Standard Deviation----------")
    for i in range(2,len(cols)):
        print(cols[i],":",data[cols[i]].std())
    print()

def plotRain(attr):
    '''
    Helper function for Q2 function. Creates a scatter plot
    x-axis: Rain
    y-axis: Other attributes
    
    Args------------------
    attr -> attribute name
    '''
    plt.scatter(data['rain'], data[attr])
    plt.xlabel("Rain")
    plt.ylabel(attr)
    plt.title("2(a) Rain vs. "+str(attr))
    plt.grid(True)
    plt.show()

def plotTemp(attr):
    '''
    Helper function for Q2 function. Creates a scatter plot
    x-axis: Temperature
    y-axis: Other attributes
    
    Args------------------
    attr -> attribute name
    '''
    plt.scatter(data['temperature'], data[attr])
    plt.xlabel("Temperature")
    plt.ylabel(attr)
    plt.title("2(b) Temperature vs. "+str(attr))
    plt.grid(True)
    plt.show()
    
def Q2():
    print("A2. Obtaining Scatter Plots...")

    # a) Others vs. Rain
    for e in cols:
        if e != 'rain' and e != 'dates' and e!= 'stationid':
            plotRain(e)

    # b) Others vs. Temperature
    for e in cols:
        if e != 'temperature' and e != 'dates' and e!= 'stationid':
            plotTemp(e)
        

def findCorr(x,y):
    '''
    Helper function for Q3. Computes correlation (Pearson)
    coefficient and prints it.

    Args-------------
    x -> first attribute name(string)
    y -> second attribute name(string)
    '''
    sumx = sum(data[x])
    sumy = sum(data[y])
    sqsumx = sum(data[x]**2)
    sqsumy = sum(data[y]**2)
    n = len(data[x])
    prodsum = 0
    for i in range(n):
        prodsum += data[x][i]*data[y][i]
    R = ((n*prodsum) - (sumx*sumy)) / math.sqrt((n*sqsumx-(sumx**2))*(n*sqsumy-(sumy**2)))
    print("Corr("+x+","+y+"):",R)
    
def Q3():
    print("A3. Finding Pearson's correlation coefficient...")
    # a) Others & Rain
    print()
    print("For rain & others:")
    for e in cols:
        if e != 'dates' and e!= 'stationid':
            findCorr('rain',e)
    # b) Others & Temperature   
    print()
    print("For temperature & others")
    for e in cols:
        if e != 'dates' and e!= 'stationid':
            findCorr('temperature',e)


def Q4():
    print("A4. Obtaining histogram (moisture vs. rain)...")
    plt.subplot(121)
    plt.hist(data['rain'], facecolor='g')
    plt.grid(True)
    plt.xlabel("Rain amount")
    plt.ylabel("Frequency")
    plt.title("Rain histogram")

    plt.subplot(122)
    plt.hist(data['moisture'])
    plt.xlabel("Moisture amount")
    plt.title("Moisture histogram")
    plt.grid(True)

    plt.suptitle("Q4 Histograms")
    plt.show()

def Q5():
    print("A5. Obtaining histogram of station-wise rain...")
    dfr = data.groupby('stationid').rain
    stids = ['t10', 't11', 't12', 't13', 't14', 't15', 't6',
             't7', 't8', 't9']
    for station in stids:
        plt.hist(dfr.get_group(station))
        plt.title(station)
        plt.xlabel("Rain at "+str(station))
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()
    
    
def Q6():
    print("A6. Obtaining box plot...")
    fig = plt.subplots()
    
    plt.subplot(121)
    plt.boxplot(data['rain'])
    plt.title("Rain")
    plt.ylabel("Frequency")
    plt.xlabel("Rain data")
    plt.grid(True)

    plt.subplot(122)
    plt.boxplot(data['moisture'])
    plt.title("Moisture")
    plt.xlabel("Moisture data")
    plt.grid(True)

    plt.show()


print("[+] Running program...")
Q1()
print()
Q2()
print("Finished successfully")
print()
Q3()
print()
Q4()
print("Finished successfully")
print()
Q5()
print("Finished successfully")
print()
Q6()
print("Finished successfully")
print("[-] -------------------- xxx --------------------")
