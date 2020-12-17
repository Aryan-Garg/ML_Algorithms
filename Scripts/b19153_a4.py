'''
Aryan Garg
B19153
+91-8219383122

DS3 - Lab Assignment 4
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

class Q1:
    '''
Q1.
Since the data is highly imbalanced,
we will have to take equal records from each class
    '''
    def __init__(self, df):
        self.X_1 = df.iloc[0:170,0:-1]
        self.y_1 = df.iloc[0:170, -1]
        
        self.X_0 = df.iloc[171:341, 0:-1]
        self.y_0 = df.iloc[171:341, -1]

        [self.X1_train,
        self.X1_test,
        self.X1_label_train,
        self.X1_label_test] = self.split(self.X_1, self.y_1)

        [self.X0_train,
         self.X0_test,
         self.X0_label_train,
         self.X0_label_test] = self.split(self.X_0, self.y_0)

        [self.df_train,
         self.df_test,
         self.df_train_label,
         self.df_test_label] = self.merge_dfs()

        self.save_files()
        self.acc_vals = self.KNN(self.df_train, self.df_train_label, self.df_test, self.df_test_label)
        
        
    def split(self,X,y):
        X_t, X_te, X_lt, X_lte = train_test_split(X,y,random_state = 42,test_size = 0.3,shuffle = True)
        return X_t, X_te, X_lt, X_lte

    def merge_dfs(self):
        df_train = (self.X0_train).append(self.X1_train, ignore_index=True)
        df_test = (self.X0_test).append(self.X1_test, ignore_index = True)

        df_train_label = (self.X0_label_train).append(self.X1_label_train, ignore_index=True)
        df_test_label = (self.X0_label_test).append(self.X1_label_test, ignore_index=True)
        return df_train, df_test, df_train_label, df_test_label

    def save_files(self):
        (self.df_train).to_csv('seismic-bumps-train.csv', index = False)
        (self.df_test).to_csv('seismic-bumps-test.csv', index = False)
        print("[+2] Created {seismic-bumps-train.csv} and {seismic-bumps-train.csv}")

    def KNN(self, train_data, train_labels, test_data, test_labels):
        print("\n*** Classifying test data using KNN for different Ks ***\n")
        from sklearn.neighbors import KNeighborsClassifier
        
        K = [1,2,3,4,5]
        acc_vals = []
        for i in K:
            knn = KNeighborsClassifier(n_neighbors = i)
            knn.fit(train_data, train_labels)
            pred = knn.predict(test_data)
            print("--------- K =",i,"---------\n")
            print("Confusion matrix:\n",confusion_matrix(test_labels, pred),"\n")
            acc_ = accuracy_score(test_labels,pred)
            print("~ Accuracy score : {0:.3f}".format(100*acc_),"%\n")
            acc_vals.append([acc_,i])

        print("-----------------------------------------")
        best_m = max(acc_vals) 
        print("Max accuracy obtained: {0:.3f} %".format(100*best_m[0]),'for k =',best_m[1])
        print("-----------------------------------------")
        return acc_vals
        
        
class Q2(Q1):
    '''
Q2.
Testing the effect of MinMax Normalization on the same model,
with the same params.
    '''
    def __init__(self, df_train_label, df_test_label):
        self.f_tr = pd.read_csv('seismic-bumps-train.csv')
        self.f_te = pd.read_csv('seismic-bumps-test.csv')
        
        self.save_norm_files()
        # Reading the new files as class instances for easy access
        self.f_normTr = pd.read_csv('seismic-bumps-train-Normalized.csv')
        self.f_normTe = pd.read_csv('seismic-bumps-test-normalized.csv')

        self.df_train_label = df_train_label
        self.df_test_label = df_test_label
        
        self.acc_vals = Q1.KNN(self, self.f_normTr, self.df_train_label, self.f_normTe, self.df_test_label)
        
        
    def save_norm_files(self):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        # MinMax scaling for training file
        ftr_ = pd.DataFrame(scaler.fit_transform(self.f_tr), columns= (self.f_tr).columns)
        ftr_.to_csv('seismic-bumps-train-Normalized.csv', index = False)

        # MinMax scaling for test file
        fte_ = pd.DataFrame(scaler.fit_transform(self.f_te), columns= (self.f_te).columns)
        fte_.to_csv('seismic-bumps-test-normalized.csv', index = False)
        print("[+2] Created {seismic-bumps-train-Normalized.csv} and {seismic-bumps-test-normalized.csv}")




class Q3:
    '''
Q3.
Creating a Bayes Classifier (Unimodal Gaussian density) for the same dataset
------
0-118   recs -> class 0
119-237 recs -> class 1

=> Prior = 0.5 for both

=>
We don't need prior to compute posterior as it'll cancel out;
we just need likelihood!
------
'''
    def __init__(self, train_labels, test_labels):
        self.train = pd.read_csv('seismic-bumps-train.csv')
        self.test = pd.read_csv('seismic-bumps-test.csv')
        self.train_labels = train_labels
        self.test_labels = test_labels
        # Initializing mean vector and covariance matrix
        
        # Training Phase...
        self.mvec_0 = self.mean_vec(0)        # mu1
        self.mvec_1 = self.mean_vec(119)      # mu2
        self.cov_0 =  self.covariance(0)      # sigma1
        self.cov_1 =  self.covariance(119)    # sigma2
        
        self.preds = [] #List of predictions
        # Testing phase...
        self.make_predictions()
        self.create_confusion()
        self.accuracy = self.calc_accuracy()
        
    def mean_vec(self, s_num):
        lst = []
        for e in self.train.columns:
            lst.append(sum((self.train)[e][s_num: s_num+118])/119)
        return np.array(lst)
            

    def covariance(self, s_num):
        M = np.cov(np.transpose(self.train[int(s_num): (int(s_num)+118)]))
        M = pd.DataFrame(M)
        M.to_csv('covariance_'+str(s_num)+'.csv') # For report
        return M
    
    def multivariate_gaussian_pdf(self,record,MU,SIGMA):
        '''
        Shapes:
            X, MU -> (p x 1)
            SIGMA -> (p x p)
        '''
        #Initialize and reshape
        record = record.reshape(-1,1)
        MU = MU.reshape(-1,1)
        p,_ = SIGMA.shape

        #Compute values
        SIGMA_inv = np.linalg.inv(SIGMA)
        denominator = np.sqrt((2 * np.pi)**p * np.linalg.det(SIGMA))
        exponent = -(1/2) * ((record - MU).T @ SIGMA_inv @ (record - MU))
    
        return float((1. / denominator) * np.exp(exponent) ) 

    def make_predictions(self):
        for i in range(len(self.test)):
            record = np.array(self.test.iloc[i])
            like_0 = self.multivariate_gaussian_pdf(record, self.mvec_0, self.cov_0)
            like_1 = self.multivariate_gaussian_pdf(record, self.mvec_1, self.cov_1)
        
            post_0 = like_0/(like_0 + like_1)
            post_1 = 1 - post_0
            if post_0 >= post_1:
                self.preds.append(0)
            else:
                self.preds.append(1)

    def calc_accuracy(self):
        print("\n-----------------")
        print("~ Accuracy of model: {0:.3f}%".format(100*accuracy_score(self.test_labels, self.preds)))
        print("-----------------")
        return accuracy_score(self.test_labels, self.preds)
        
    def create_confusion(self):
        print("Confusion matrix for the Bayes classifier:\n",confusion_matrix(self.test_labels, self.preds))
        

class Q4:
    '''
Q4.
Comparing...
(i.)   K- Nearest Neighbour(s) classifier
(ii.)  MinMax Normalized data on the same classifer (i)
(iii.) Baye's Classifier for Unimodal Gaussian Density           
    '''

    def __init__(self, q1_acc, q2_acc, q3_acc):
        self.q1_acc = q1_acc
        self.q2_acc = q2_acc
        self.q3_acc = q3_acc
        print("\n(Q1)KNN model stats:\n")
        self.best_q1 = self.create_table(self.q1_acc)
        print("\n(Q2)Normalized data KNN model stats:\n")
        self.best_q2 = self.create_table(self.q2_acc)
        print("\nFinal Comparison:\n")
        self.compare_all()

    def create_table(self, acc_k):
        print("{0:20} {1:20}".format("Value of K", "Accuracy (in %)"))
        for e in acc_k:
            print("{0:5d} {1:25f}%".format(e[1], e[0]*100))
        print("------------------------------------------------------------------")
        return max(acc_k)
    
    def compare_all(self):
        print("------------------------------------------------------------------")
        print("{0:25} {1:30}".format("Classifier Model" ,"Best Score\n"))
        print("{0:23} {1:30}".format("1. Q1 KNN", "~= "+str(int(100*self.best_q1[0]))+"% for K = "+str(self.best_q1[1])))
        print("{0:23} {1:30}".format("2. Q2 KNN", "~= "+str(int(100*self.best_q2[0]))+"% for K = "+str(self.best_q2[1])))
        print("{0:23} {1:30}".format("3. Bayes Classifier", str(100*self.q3_acc)+"%"))
        print()
        print("[+]Conclusion:\nBayes classifier gave the best results!")
        print("------------------------------ xxx ------------------------------")
        print("\n[-]Finished with exit value 0")
        
        

    
data = pd.read_csv("seismic_bumps1.csv")
data.drop(['nbumps', 'nbumps2','nbumps3','nbumps4','nbumps5','nbumps6','nbumps7','nbumps89'], inplace = True,axis=1)
df = data.sort_values(by = 'class', ascending = False) #sorting records by class value

print('''
Assignment - 4
Data Classification using KNN and Bayes Classifier for Unimodal Gaussian Density
''')
print("Checking # of records per class...")
print("_________________________")
print("Class: #Records")
print(df['class'].value_counts())
print("_________________________")

print(Q1.__doc__)
obj1 = Q1(df)
print("_________________________________________________________________________")

print(Q2.__doc__)
obj2 = Q2(obj1.df_train_label, obj1.df_test_label)
print("_________________________________________________________________________")

print(Q3.__doc__)
obj3 = Q3(obj1.df_train_label, obj1.df_test_label)
print("_________________________________________________________________________")

print(Q4.__doc__)
Q4(obj1.acc_vals, obj2.acc_vals, obj3.accuracy)

