'''
Aryan Garg
B19153
+91-8219383122

Lab Assignment - 3
'''
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

class Lab_3:
    class fun:
        def get_outliers(self,df):
            minimum = 2.5 * np.percentile(df, 25) - 1.5 * np.percentile(df, 75)
            maximum = 2.5 * np.percentile(df, 75) - 1.5 * np.percentile(df, 25)
            return pd.concat((df[df < minimum], df[df > maximum]))

        def replace_outliers(self,column):
            for i in column:
                while len(self.get_outliers(self.df[i])):
                    self.df[i][self.get_outliers(self.df[i]).index]=self.df[i].median()

        def MSE(self,a,b):
            e=0
            for i,j in zip(a,b):
                e+=(i-j)**2
            return e/len(a)

        def RMSE(self,df,re):
            RMSE=[]
            for i in range(len(df.columns)):
                MSE = self.MSE(df.iloc[:,i],re[:,i])
                RMSE.append(MSE**0.5)
            return RMSE

    class Q1(fun):
        def __init__(self,filepath="landslide_data3.csv"):
            self.df=pd.read_csv(filepath)
            self.replace_outliers(self.df.columns[2:])

        def Q1a(self,a,b,comment=True):
            d={"BEFORE (Min, Max)":[],"AFTER (Min, Max)":[]}
            df=self.df.copy(deep=comment)
            for i in df.columns[2:]:
                minimum=min(df[i])
                maximum=max(df[i])
                d["BEFORE (Min, Max)"].append((minimum,maximum))
                for j in range(len(df[i])):
                    df[i].iloc[j] = a + ((df[i].iloc[j]-minimum)/(maximum-minimum))*(b-a)
                d["AFTER (Min, Max)"].append((min(df[i]), max(df[i])))
            return pd.DataFrame(d,index=df.columns[2:])

        def Q1b(self,comment=True):
            d = {"BEFORE (μ, σ)": [], "AFTER (μ, σ)": []}
            df = self.df.copy(deep=comment)
            for i in df.columns[2:]:
                mue = np.mean(df[i])
                sig = np.std(df[i])
                d["BEFORE (μ, σ)"].append((mue, sig))
                for j in range(len(df[i])):
                    df[i].iloc[j] = (df[i].iloc[j] - mue) / sig
                d["AFTER (μ, σ)"].append((np.mean(df[i]), np.std(df[i])))
            return pd.DataFrame(d, index=df.columns[2:])

        def show(self):
            print("Q1")
            print("a) Min-Max Normalization:\n")
            print("          Attributes"+str(self.Q1a(3,9)).replace("\n","\n\t\t"))
            print("\nb) Standardization with μ=0, σ=1:")
            print("\t"+str(self.Q1b()).replace("\n","\n\t"))
            print("--------------------------------------------------------------------------------------")

    class Q2(fun):
        def __init__(self):
            self.matrix=np.array([[6.84806467,7.63444163],[7.63444163,13.020774623]])
            self.data=np.random.multivariate_normal([0,0],self.matrix,1000,check_valid="ignore")
            self.evals, self.evect = np.linalg.eig(np.cov(self.data.transpose()))
            self.prj=np.dot(self.data,self.evect)

        def Q2a(self,comment=True):
            plt.title("Scatter plot of data")
            plt.scatter(self.data[:,0],self.data[:,1],marker="x",alpha=0.5)
            plt.xlabel("Sample [0]")
            plt.ylabel("Sample [1]")
            if comment:
                plt.show()

        def Q2b(self):
            evals, evect = self.evals, self.evect
            print("\t\tEigenvalues: ",end="")
            print(*evals,sep=", ")
            print("\t\tEigenvectors: ",end="")
            print(*evect.transpose(),sep=", ")
            self.Q2a(False)
            for x,y in evect.transpose():
                plt.quiver(0,0,x,y,angles="xy",scale=3,color="red")
            plt.title("2D synthetic data and eigen directions")
            plt.show()

        def Q2c(self):
            evals, evect = self.evals, self.evect
            fig,ax = plt.subplots(1,2,figsize=(12,6))
            prj=self.prj
            for i in range(2):
                X = evect[0][i] * prj[:, i]
                Y = evect[1][i] * prj[:, i]
                ax[i].scatter(self.data[:, 0], self.data[:, 1], marker="x", alpha=0.5)
                for x, y in evect.transpose():
                    ax[i].quiver(0, 0, x, y, angles="xy",scale=3, color="red")
                ax[i].scatter(X,Y,marker="x",alpha=0.5,color="orange")
                ax[i].set_xlabel("Sample [0]")
                ax[i].set_ylabel("Sample [1]")
            ax[0].set_title("Projected values onto the first eigen")
            ax[1].set_title("Projected values onto the second eigen")
            fig.suptitle("Projection of data onto Eigenvectors",fontsize=18)
            plt.show()

        def Q2d(self):
            self.re_data = np.dot(self.evect,self.prj.transpose())
            print(f"\t\tMSE for Sample [0]: {self.MSE(self.re_data[0], self.data.transpose()[0])}")
            print(f"\t\tMSE for Sample [1]: {self.MSE(self.re_data[1], self.data.transpose()[1])}")

        def show(self):
            print("Q2")
            print("\na) Scatter plot of data:\n")
            self.Q2a()
            print("b) Eigenvalues and Eigenvectors of covariance matrix:\n")
            self.Q2b()
            print("\nc) Projection of data onto Eigenvectors:\n")
            self.Q2c()
            print("\nd) Reconstruction error between reconstructed data and original data:\n")
            self.Q2d()
            print("--------------------------------------------------------------------------------------")

    class Q3(Q1,fun):
        def __init__(self,filepath="landslide_data3.csv"):
            super().__init__(filepath)
            self.Q1b(True)

        def Q3a(self):
            self.pca = PCA(n_components=2).fit_transform(self.df.iloc[:, 2:])
            evals = np.linalg.eigvals(np.cov(self.pca.transpose()))
            print(f"\tVariance, Eigenvalue at Dimension [0]: {np.var(self.pca[:, 0])}, {evals[0]}")
            print(f"\tVariance, Eigenvalue at Dimension [1]: {np.var(self.pca[:, 1])}, {evals[1]}")
            plt.scatter(self.pca[:,0],self.pca[:,1],marker="x",alpha=0.5)
            plt.title("Scatter plot of reduced dimensional data")
            plt.xlabel("Dimension [0]")
            plt.ylabel("Dimension [1]")
            plt.show()

        def Q3b(self):
            evals = sorted(np.linalg.eigvals(np.cov(self.df.iloc[:,2:].transpose())),reverse=True)
            plt.stem(range(7),evals)
            plt.yscale("log")
            plt.title("Eigenvalues in descending order")
            plt.xlabel("Index")
            plt.ylabel("Eigenvalue")
            plt.show()

        def Q3c(self):
            l=[]
            org = self.df.iloc[:, 2:]
            for i in range(1,len(org.columns)+1):
                pca = PCA(n_components=i)
                com = pca.fit_transform(org)
                re = pca.inverse_transform(com)
                l.append(self.RMSE(org,re))
            l=np.array(l).transpose()
            fig, ax = plt.subplots(3, 3, figsize=(12, 9))
            r, c = 0, 0
            for i in range(len(org.columns)):
                ax[r,c].bar(range(1,8),l[i])
                ax[r,c].set_title(org.columns[i].title())
                ax[r,c].set_xlabel("Value of l")
                ax[r,c].set_ylabel("RMSE")
                c+=1
                if c==3:
                    r+=1;c=0
                    if r==2: c=1
        
            fig.delaxes(ax[2,0])
            fig.delaxes(ax[2,2])
            fig.tight_layout()
            plt.show()

        def show(self):
            print("Q3")
            print("\na) Variance and Eigenvalue of projected data:\n")
            self.Q3a()
            print("\nb) Plot of Eigenvalues:\n")
            self.Q3b()
            print("\nc) Plot of Reconstruction errors in terms of RMSE:\n")
            self.Q3c()

    def show(self):
        self.Q1().show()
        self.Q2().show()
        self.Q3().show()
        print("--------------------------------------------------------------------------------------")

print("\t\t\t***** Lab Assignment - 3 *****\n")
Lab_3().show()
print("[+] Executed successfully")
