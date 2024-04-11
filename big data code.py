import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dask import dataframe as dd
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from dask_ml.model_selection import train_test_split
import dask_ml.model_selection as dcv

import joblib

distance = pd.read_csv("Trips_by_DistanceFinal.csv")
df = pd.read_csv("Trips_Full DataFinal.csv")

#set date column to a date datatype
distance['Date'] = pd.to_datetime(distance['Date'])
df['Date'] = pd.to_datetime(df['Date'])

#filter to national
distance.loc[distance['Level'] == "National"]

def exploreData(df, distance):
    # check the top of the data with pandas
    print(df.head())  
    print(distance.head())

    # check the bottom of the data with pandas
    print(df.tail())
    print(distance.tail())

    print(" ")
    #summary of the data
    print(df.info())
    print("")
    print(distance.info()) 

def oneAButTimed():
    #time the how long it takes to run the code
    start_time = time.time()

    distance = pd.read_csv("Trips_by_DistanceFinal.csv")

    #set date column to a date datatype
    distance['Date'] = pd.to_datetime(distance['Date'])


    #filter to national
    task_1a = distance.loc[distance['Level'] == "National"]

    #group date colum to week
    task_1a = task_1a.groupby(pd.Grouper(key = 'Date', freq = 'W'))

    #calculate average
    task_1a = task_1a['Population Staying at Home'].mean()


    # Stop the timer
    totalTime = time.time() - start_time

    # Print the elapsed time
    print("Time taken:", totalTime, "seconds")


def oneBButTimed():
    #time the how long it takes to run the code
    start_time = time.time()

    distance = pd.read_csv("Trips_by_DistanceFinal.csv")

    #set date column to a date datatype
    distance['Date'] = pd.to_datetime(distance['Date'])

    #filter to national
    distance.loc[distance['Level'] == "National"]
    #1ai) How many people are staying at home? 


    #filter columns
    distance10to25 = distance.loc[distance['Number of Trips 10-25'] > 10000000]
    distance50to100 = distance.loc[distance['Number of Trips 50-100'] > 10000000]

        
    # Stop the timer
    totalTime = time.time() - start_time

    # Print the elapsed time
    print("Time taken:", totalTime, "seconds")




def oneA():
    #####################################################################################
    #1ai) How many people are staying at home? 

    #set date column to a date datatype
    distance['Date'] = pd.to_datetime(distance['Date'])

    #filter to national
    task_1a = distance.loc[distance['Level'] == "National"]

    #group date colum to week
    task_1a = task_1a.groupby(pd.Grouper(key = 'Date', freq = 'W'))

    #calculate average
    task_1a = task_1a['Population Staying at Home'].mean().round().reset_index()

    #add week column
    task_1a['Week'] = (task_1a.index+1).astype(str)

    #print week and population staying at home (might want to change to week)
    task_1a = task_1a[['Week', 'Population Staying at Home']]
    print(task_1a)

    #plot the data in task_1a
    task_1a['Week'] = task_1a['Week'].astype(int)
    plt.plot(task_1a['Week'], task_1a['Population Staying at Home'])
    plt.xlabel('Week')
    plt.ylabel('Population Staying at Home')
    plt.title('Population Staying at Home per Week')
    plt.tight_layout()
    plt.show()

    ##############################################################################
    #1aii) 
    distanceCategories = ['Trips <1 Mile', 'Trips 1-3 Miles', 'Trips 3-5 Miles', 'Trips 5-10 Miles', 
                        'Trips 10-25 Miles', 'Trips 25-50 Miles', 'Trips 50-100 Miles', 
                        'Trips 100-250 Miles', 'Trips 250-500 Miles', 'Trips 500+ Miles']

    totalDistance = df[distanceCategories].sum()
    print(totalDistance)

    # Plotting
    plt.bar(distanceCategories, totalDistance, color='skyblue')

    # Add labels and title
    plt.title('Total Number of Trips by Distance Category')
    plt.xlabel('Distance Category')
    plt.ylabel('Total Number of Trips')

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()

def oneB(distance):
    #filter to national
    distance = distance.loc[distance["Level"] == "National"]

    #convert distance to datetime 
    distance['Date'] = pd.to_datetime(distance['Date'])

    #filter columns
    distance10to25 = distance.loc[distance['Number of Trips 10-25'] > 10000000]
    distance50to100 = distance.loc[distance['Number of Trips 50-100'] > 10000000]

    print(distance10to25['Number of Trips 10-25'])
    print(distance50to100['Number of Trips 50-100'])

    #create layout for 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Add labels and titles for each subplot
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Number of Trips 10-25")
    axes[0].set_title('Number of Trips 10-25 by Date')

    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Number of Trips 50-100")
    axes[1].set_title('Number of Trips 50-100 by Date')   

    # Plot for distance10to25
    axes[0].scatter(distance10to25['Date'], distance10to25['Number of Trips 10-25'], label='Number of Trips 10-25')
    
    # Plot for distance50to100
    axes[1].scatter(distance50to100['Date'], distance50to100['Number of Trips 50-100'], label='Number of Trips 50-100')

    # make a bigger gap between the subplots
    plt.subplots_adjust(hspace=0.5) 

    # Show the plot
    plt.show()

def oneC(distance):
    # Define number of processors
    n_processors = [10, 20]
    n_processors_time = {}  # Define n_processors_time dictionary

    #1aii
    # Loop over each processor configuration
    for processor in n_processors:
        start_time = time.time()

        
        # Set date column to a date datatype
        distance = dd.read_csv('Trips_by_DistanceFinal.csv',
                                   dtype={'County Name': 'object',
                                          'Number of Trips': 'float64',
                                          'Number of Trips 1-3': 'float64',
                                          'Number of Trips 10-25': 'float64',
                                          'Number of Trips 100-250': 'float64',
                                          'Number of Trips 25-50': 'float64',
                                          'Number of Trips 250-500': 'float64',
                                          'Number of Trips 3-5': 'float64',
                                          'Number of Trips 5-10': 'float64',
                                          'Number of Trips 50-100': 'float64',
                                          'Number of Trips <1': 'float64',
                                          'Number of Trips >=500': 'float64',
                                          'Population Not Staying at Home': 'float64',
                                          'Population Staying at Home': 'float64',
                                          'State Postal Code': 'object'})
        
        # Convert date column to datetime
        distance['Date'] = dd.to_datetime(distance['Date'])

        # Filter to national
        task_1a = distance.loc[distance['Level'] == "National"]


        # Group by week
        task_1a['Date'] = task_1a['Date'].dt.strftime('%y %W')

        # Group by week and calculate the mean
        temp = task_1a.groupby('Date')['Population Staying at Home'].mean()

        temp = temp.compute(num_workers = processor)
        #print("task_1a")
        #print(task_1a)

        dask_time = time.time() - start_time
        n_processors_time[processor] = dask_time

    print(n_processors_time)

    #1aii
    """
    for processor in n_processors:
        start_time = time.time()


        distanceCategories = ['Trips <1 Mile', 'Trips 1-3 Miles', 'Trips 3-5 Miles', 'Trips 5-10 Miles', 
                        'Trips 10-25 Miles', 'Trips 25-50 Miles', 'Trips 50-100 Miles', 
                        'Trips 100-250 Miles', 'Trips 250-500 Miles', 'Trips 500+ Miles']
        
        totalDistance = df[distanceCategories].mean()

        totalDistance = dd.compute(totalDistance)


        dask_time = time.time() - start_time
        n_processors_time[processor] = dask_time

    print(n_processors_time)
    """
    
    #1b
    for processor in n_processors:
        start_time = time.time()

        distance = dd.read_csv('Trips_by_DistanceFinal.csv',
                                   dtype={'County Name': 'object',
                                          'Number of Trips': 'float64',
                                          'Number of Trips 1-3': 'float64',
                                          'Number of Trips 10-25': 'float64',
                                          'Number of Trips 100-250': 'float64',
                                          'Number of Trips 25-50': 'float64',
                                          'Number of Trips 250-500': 'float64',
                                          'Number of Trips 3-5': 'float64',
                                          'Number of Trips 5-10': 'float64',
                                          'Number of Trips 50-100': 'float64',
                                          'Number of Trips <1': 'float64',
                                          'Number of Trips >=500': 'float64',
                                          'Population Not Staying at Home': 'float64',
                                          'Population Staying at Home': 'float64',
                                          'State Postal Code': 'object'})   

        #filter to national
        distance = distance.loc[distance["Level"] == "National"]

        #convert distance to datetime 
        distance['Date'] = dd.to_datetime(distance['Date'])

        #filter columns
        distance10to25 = distance.loc[distance['Number of Trips 10-25'] > 10000000]
        distance50to100 = distance.loc[distance['Number of Trips 50-100'] > 10000000]

        distance10to25 = distance10to25.compute(num_workers = processor)
        distance50to100 = distance50to100.compute(num_workers = processor)
        

        dask_time = time.time() - start_time
        n_processors_time[processor] = dask_time

    print(n_processors_time)

def oneD(distance, df):
    ##################################Linear Regression############################################
    #filter to national
    distance = distance.loc[distance["Level"] == "National"]
    df = df.loc[df["Level"] == "National"]  

    #convert distance to datetime 
    distance['Date'] = pd.to_datetime(distance['Date'])


    #filter columns to week 31 of 2019
    week31 = distance[distance['Week'] == 31]
    week31_2019 = week31[(week31['Date'].dt.year == 2019) & (week31['Week'] == 31)]

    #define x and y
    y = week31_2019[['Number of Trips 10-25']]
    x = df[['Trips 1-25 Miles']]

    print("x and y")
    print(x)
    print(y)

    print("=================")
    print("Linear Regression")
    print("=================")

    #make the linear model of x and y
    model = LinearRegression()
    model.fit(x, y)

    #get the r squared value, intercept, and coefficients of the linear regression model
    rSquared = model.score(x, y)
    intercept = model.intercept_
    coefficients = model.coef_
    print("R Squared Value: ", rSquared)
    print("intercept: ", intercept)
    print("coefficients: ", coefficients)

    #predict the response variable y based off of x
    y_pred = model.predict(x)
    print(f"predicted response:", y_pred)

    ##################################modelling################################################################
    print("=================")
    print("Linear Regression Model")
    print("=================")

    x = x.to_numpy()
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle = False)

    print("X_train")
    print(X_train)
    print(" ")
    print("X_test")
    print(X_test)
    print(" ")
    print("y_train")
    print(y_train)
    print(" ")
    print("y_test")
    print(y_test)
    print(" ")

    rSquaredModel = model.score(X_train, y_train)
    print("R Squared Value: ", rSquared)

    ##################################Graph of Linear Regression############################################
    #plot the ine of best fit
    fig, ax = plt.subplots()

    # y = mx + c to find y at x[0][0]
    tmp = coefficients[0][0] * x[0] + intercept[0]

    # line from (x[0][0], tmp) with slope of model slope
    ax.axline((x[0][0], tmp[0]), slope = coefficients[0][0], color='black', label='Line of Best Fit')

    #plot x and y and x and y_pred on the smae graph
    ax.scatter(x, y)
    ax.scatter(x, y_pred)
    plt.xlabel("Trips 1-25 Miles")
    plt.ylabel("Number of Trips 10-25")
    plt.title("Trips 1-25 Miles vs Number of Trips 10-25")

    plt.show()

    ##################################Multiple Linear Regression############################################
    print("==========================")
    print("Multiple Linear Regression")
    print("==========================")

    y = week31_2019[['Number of Trips 10-25']]
    x = df[['Trips 1-25 Miles', 'Trips 25-100 Miles']]

    #make the linear model of x and y
    model = LinearRegression()
    model.fit(x, y)

    #get the r squared value, intercept, and coefficients
    rSquared = model.score(x, y)
    intercept = model.intercept_
    coefficients = model.coef_
    print("R Squared Value: ", rSquared)
    print("intercept: ", intercept)
    print("coefficients: ", coefficients)

    y_pred = model.predict(x)
    print(f"predicted response:", y_pred)

    ##################################Polynomial Regression#################################################
    print("=====================")
    print("Polynomial Regression")
    print("=====================")

    y = week31_2019[['Number of Trips 10-25']]

    x = df[['Trips 1-25 Miles']]

    # Initialize the degree of the polynomial features object
    polyFeatures = PolynomialFeatures(degree=2)

    # fit x to the polynomial features object
    xPoly = polyFeatures.fit_transform(x)

    # Initialize the linear regression model
    model = LinearRegression()

    # Fit the model
    model.fit(xPoly, y)
    
    # Get the R-squared value, intercept, and coefficients
    r_squared = model.score(xPoly, y)
    intercept = model.intercept_
    coefficients = model.coef_
    print("R Squared Value:", r_squared)
    print("Intercept:", intercept)
    print("Coefficients:", coefficients)

    # get the predicted response
    y_pred = model.predict(xPoly)
    print("Predicted response:", y_pred)





def oneE(df):
    # Extracting trip distance columns
    tripDistanceCol = ['Trips <1 Mile', 'Trips 1-3 Miles', 'Trips 3-5 Miles', 'Trips 5-10 Miles', 
                        'Trips 10-25 Miles', 'Trips 25-50 Miles', 'Trips 50-100 Miles', 
                        'Trips 100-250 Miles', 'Trips 250-500 Miles', 'Trips 500+ Miles']


    dates = df['Date'].dt.date
    tripData = df[tripDistanceCol]
    print(dates)
    print(tripData)


    # Plotting bar plot
    tripData.plot(kind='bar', width = 1)

    # Add labels and title
    plt.title('Total Number of Trips by Date')
    plt.xlabel('Date')
    plt.ylabel('Total Number of Trips')

    # Rotate x-axis labels
    plt.xticks(range(len(dates)), dates, rotation=45)

    # Show plot
    plt.tight_layout()
    plt.show()

    #####################################################################################
    # Plotting line plot
    plt.plot(dates, tripData, label=tripDistanceCol)

    # Adding labels and title
    plt.title('Total Number of Trips by Date')
    plt.xlabel('Date')
    plt.ylabel('Total Number of Trips')

    # Rotate x-axis labels and add legend
    plt.xticks(rotation=45)
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.show()


#exploreData(df, distance)
#oneAButTimed()
#oneBButTimed()
oneA()
oneB(distance)
#oneC(distance)
#oneD(distance, df)
oneE(df)