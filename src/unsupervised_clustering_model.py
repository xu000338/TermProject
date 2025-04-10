import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import kmeans model
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
 
import warnings
warnings.filterwarnings("ignore")
  
#import logger module
from .logger import exception_decorator

@exception_decorator
def load():

    df = pd.read_csv('./data/mall_customers.csv')
    # Let' train our model on spending_score and annual_income
    kmodel = KMeans(n_clusters=5).fit(df[['Annual_Income','Spending_Score']])

    # check your cluster centers
    kmodel.cluster_centers_

    # Put this data back in to the main dataframe corresponding to each observation
    df['Cluster'] = kmodel.labels_
    # Let' visualize these clusters
    sns.scatterplot(x='Annual_Income', y = 'Spending_Score', data=df, hue='Cluster', palette='colorblind')
    plt.savefig('./images/clusters.png')
    return df
    
@exception_decorator
def elbow_plot(df):
        
        #elbow method
        # try using a for loop
        k = range(3,9)
        K = []
        WCSS = []
        for i in k:
            kmodel = KMeans(n_clusters=i).fit(df[['Annual_Income','Spending_Score']])
            wcss_score = kmodel.inertia_
            WCSS.append(wcss_score)
            K.append(i)
        
        # Store the number of clusters and their respective WSS scores in a dataframe
        wss = pd.DataFrame({'cluster': K, 'WSS_Score':WCSS})
        # Now, plot a Elbow plot
        wss.plot(x='cluster', y = 'WSS_Score')
        plt.xlabel('No. of clusters')
        plt.ylabel('WSS Score')
        plt.title('Elbow Plot')
        plt.savefig('./images/elbow_plot.png')

@exception_decorator
def silhouette(df):
    #silhouette method
    # same as above, calculate sihouette score for each cluster using a for loop

    # try using a for loop
    k = range(3,9) # to loop from 3 to 8
    K = []         # to store the values of k
    ss = []        # to store respective silhouetter scores
    WCSS=[]
    
    for i in k:
        kmodel = KMeans(n_clusters=i,).fit(df[['Annual_Income','Spending_Score']], )
        ypred = kmodel.labels_
        wcss_score = kmodel.inertia_
        WCSS.append(wcss_score)
        sil_score = silhouette_score(df[['Annual_Income','Spending_Score']], ypred)
        K.append(i)
        ss.append(sil_score)
    
    # Store the number of clusters and their respective silhouette scores in a dataframe
    wss = pd.DataFrame({'cluster': K, 'WSS_Score':WCSS})

    wss['Silhouette_Score']=ss
      
    # Now, plot the silhouette plot
    wss.plot(x='cluster', y='Silhouette_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Plot')
    plt.savefig('./images/silhouette.png')
    
@exception_decorator
def k_means_model(df):
    
    # Train a model on 'Age','Annual_Income','Spending_Score' features
    k = range(3,9)
    K = []
    ss = []
    for i in k:
        kmodel = KMeans(n_clusters=i).fit(df[['Age','Annual_Income','Spending_Score']], )
        ypred = kmodel.labels_
        sil_score = silhouette_score(df[['Age','Annual_Income','Spending_Score']], ypred)
        K.append(i)
        ss.append(sil_score)
    
    Variables3 = pd.DataFrame({'cluster': K, 'Silhouette_Score':ss})
    Variables3.plot(x='cluster', y='Silhouette_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Plot With 3 Features')
    plt.savefig('./images/Variables3.png')


@exception_decorator
def main():
    df = load()
    elbow_plot(df)
    silhouette(df)
    k_means_model(df)
    
if __name__ == "__main__":
    main()