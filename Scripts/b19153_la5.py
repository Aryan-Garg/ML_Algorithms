'''
Aryan Garg
B19153
Lab Assignment 5
'''

# Modules 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

np.random.seed(10)

# Classifiers and Regressors
class partA:
    '''
    PART A:
    1. Bayes classifier with multi-modal GMM with Q modes
    2. Compare with assignment 4 models
    '''
    def __init__(self, df_tr, df_te, df_trL, df_teL):
        self.df_train = df_tr
        self.df_test = df_te
        self.df_train_labels = df_trL
        self.df_test_labels = df_teL
        
        self.Q = [2,4,8,16]
        self.acc_scores = []

        print("1 a) Building a GMM...")
        for components in self.Q:
            self.acc_scores.append([self.GausMix(components),components])
            
        self.bestAcc, self.bestK = self.bestAccGMM()
        
        print("\n2. Comparision of all the models made so far...\n")
        self.compareAll()
        
            
    def GausMix(self, comps):
        from sklearn.mixture import GaussianMixture
        GMM = GaussianMixture(n_components=comps, covariance_type='full')

        GMM.fit(self.df_train, self.df_train_labels)
        y_preds = GMM.predict(self.df_test)
        wted_lg_probs = GMM.score_samples(self.df_train)
        print("      K =",comps)
        print("\nConfusion matrix:\n")
        print(confusion_matrix(self.df_test_labels, y_preds))
        
        acc = (accuracy_score(self.df_test_labels, y_preds)*100)
        print('\n-------------------------')
        print("Accuracy score: {0:.3f}%".format(acc))
        print('-------------------------')
        print()
        
        return acc

    def bestAccGMM(self):
        print("1 b)")
        print('[+] Results...\n')
        acc, k = max(self.acc_scores)
        print("Best accuracy: {0:.3f}%".format(acc))
        print("For components:",k)
        print("_______________________")
        return acc, k
    
    def compareAll(self):
        classifiers = ['KNN','KNN on normalized data','Bayes - Unimodal', 'Bayes - GMM']
        accuracies = [75.490, 72.549, 78.431, self.bestAcc]
        print(" _________________________________________________")
        print("\n      {0:30}{1:15}".format("Classifiers", "Accuracy in %"))
        print(" _________________________________________________")
        
        for i in range(4):
            print("| {0}   {1:25} : {2:10.3f}%     |".format(i+1, classifiers[i], accuracies[i]))
        print("|_________________________________________________|")
        
        

class partB:
    '''
    PART B:
    1. Simple Linear Regressor (pressure vs. temperature)
    2. Simple Non-Linear Regressor using polynomial curve fitting
        (given pressure,predict temperature)
    '''
    def __init__(self,df):
        self.df = df
        self.X = atm_data.iloc[0:, 0:-1]
        self.y = atm_data['temperature']
        self.X_train, self.X_test, self.y_train, self.y_test = self.split(self.X, self.y)

        self.saveFiles()

        self.linearRegressor() 
        self.nonlinearRegressor()
        
    def split(self,X,y):
         return train_test_split(X, y, test_size=0.3, random_state=42)

    def saveFiles(self):
        self.X_train.to_csv('atmosphere-train.csv',index=False)
        self.X_test.to_csv('atmosphere-test.csv',index=False)
        self.y_test.to_csv('atmosphere-test-labels.csv',index=False,header=False)
        self.y_train.to_csv('atmosphere-train-labels.csv',index=False,header=False)

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
        
    def linearRegressor(self):
        regressor = LinearRegression()
        x = np.array(self.X_train['pressure']).reshape(-1,1)
        x_test = np.array(self.X_test['pressure']).reshape(-1, 1)
        
        regressor.fit(x, self.y_train)
        y_pred = regressor.predict(x_test)
        y_pred_train = regressor.predict(x)
        print("1.")
        # a)
        plt.title("Linear Regression Pressure vs. Temperature")
        plt.scatter(x_test, self.y_test, color = 'r')
        plt.plot(x_test, y_pred, lw = 3)
        plt.xlabel('Pressure')
        plt.ylabel('Temperature')
        plt.grid(True)
        plt.show()

        RMSE_train = self.rmse(y_pred_train, self.y_train)
        pred_acc_train = 1.96 * RMSE_train
        print("b) Prediction accuracy of train data: {0:.3f}".format(pred_acc_train))
        
        RMSE_test = self.rmse(y_pred, self.y_test)
        pred_acc_test = 1.96 * RMSE_test
        print("c) Prediction accuracy of test data: {0:.3f}".format(pred_acc_test))

        # d)
        plt.title('Actual vs. Predicted temperature')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True)
        plt.scatter(self.y_test, y_pred)
        plt.show()

    def plot_bar(self, p, p_err):
        for e in p_err:
            print(e)
            
        plt.title("2. Bar plot of RMSE for different Ps")
        plt.grid(True)
        plt.bar(p, p_err, color = 'r')
        plt.xlabel("P value")
        plt.ylabel("RMSE value")
        plt.show()

    
    def nonlinearRegressor(self):
        x = np.array(self.X_train['pressure']).reshape(-1,1)
        y_train = np.array(self.y_train).reshape(-1, 1)
        x_test = np.array(self.X_test['pressure']).reshape(-1,1)
        y_test = np.array(self.y_test).reshape(-1,1)
        
        # a) Over training data
        p = [2,3,4,5]
        p_rmse = []

        y_pred_train = []
        
        for e in p:
            polynomial_features = PolynomialFeatures(degree = e)
            x_poly = polynomial_features.fit_transform(x)
            
            polyregressor = LinearRegression()

            polyregressor.fit(x_poly, y_train)
            y_pred = polyregressor.predict(x_poly)
            
            p_rmse.append(1.96 * self.rmse(y_pred, y_train))

            if e == 5:
                y_pred_train = y_pred
                
            
        self.plot_bar(p, p_rmse)
        
        p_rmse2 = []
        
        y_pred_test = []
        for e in p:
            polynomial_features = PolynomialFeatures(degree = e)

            x_poly = polynomial_features.fit_transform(x)
            x_poly_test = polynomial_features.fit_transform(x_test)
            
            polyregressor2 = LinearRegression()

            polyregressor2.fit(x_poly, y_train)
            y_pred_ = polyregressor2.predict(x_poly_test)

            if e == 5:
                y_pred_test = y_pred_
                
            p_rmse2.append(1.96 * self.rmse(y_pred_, y_test))

        self.plot_bar(p, p_rmse2)
       
        # c) p = 5 gave lowest RMSE on test data
        plt.title("B2cBest fit curve on training data")
        plt.xlabel("Pressure")
        plt.ylabel("Temperature")
        plt.grid(True)
        plt.scatter(x, y_train)
        plt.plot(x, y_pred_train, color='g')
        plt.show()
        
        # d) predicted & actual temp. scatter plot
        plt.title("B2d Actual vs. Predicted Temperature")
        plt.ylabel("Predicted")
        plt.xlabel("Actual")
        plt.grid(True)
        plt.scatter(y_test, y_pred_test)
        plt.show()
        

print('''
                                    Lab Assignment - 5
                    
                    Data Classification using Bayes Classifier with GMM
                                            &
            Regression using Simple Linear Regression and Polynomial Curve Fitting
''')

# Data files
train_data = pd.read_csv('seismic-bumps-train.csv')
train_labels = pd.read_csv('train_labels.csv')
test_data = pd.read_csv('seismic-bumps-test.csv')
test_labels = pd.read_csv('test_labels.csv', index_col=0)

atm_data = pd.read_csv('atmosphere_data.csv')

print("-------------------------------------------------------------------------------------")

print(partA.__doc__)
partA(train_data, test_data, train_labels, test_labels)

print(partB.__doc__)
partB(atm_data)

print("------------------------------------------xxxxx--------------------------------------")
