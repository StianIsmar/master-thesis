from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import DBSCAN
import seaborn as sns
from matplotlib import cm
from sklearn.cluster import KMeans
from scipy import stats
import sys, os,os.path

ROOT_PATH = os.path.abspath("..").split("data_processing")[0]
module_paths = []
module_paths.append(os.path.abspath(os.path.join(ROOT_PATH+"/data_processing/")))
module_paths.append(os.path.abspath(os.path.join(ROOT_PATH+"/hybrid_analysis_process_functions/")))
module_paths.append(os.path.abspath(os.path.join(ROOT_PATH+"/utils/")))
for module_path in module_paths:
    if module_path not in sys.path:
        print("appended")
        sys.path.append(module_path)
import functions as f
c, p = f.color_palette()

sns.set(context='paper', style='whitegrid', palette=np.array(p))
ROOT_PATH = os.path.abspath(".").split("src")[0]

plt.style.use('file://' + ROOT_PATH + "src/utils/plotparams.rc")

def plot_clusters(y_cluster_kmeans,feature_df,plot=True):
    '''
    Args:
        y_cluster_kmeans (List): List of all datapoints and which cluster they are assigned to.
        feature_df (pd.DataFrame): Feature DataFrame for all vibration intervals features for a WT.
    Returns: 
        clusterDict: (dictionary) Dictionary with keys=clusters and values=indexes. 
    '''
    def map_index(index):
        return feature_df.iloc[index,:]['Index']


    myDict = {}
    min_clus = y_cluster_kmeans.min()
    
    if not (min_clus == 0):
        if min_clus < 0:
            y_cluster_kmeans = y_cluster_kmeans + (min_clus*-1 )    
        elif min_clus > 0:
            y_cluster_kmeans = y_cluster_kmeans - min_clus
        min_clus = 0
    
    # Placing the clustered points in cluster lists (where the lists in each key contain indexes)
    for i, elem in enumerate(y_cluster_kmeans):
        if (elem in myDict):
            myDict[elem].append(i)
        else:
            myDict[elem] = [i]

    for i in range(min_clus,len(myDict)): # Sorting the dictionaries
        np.sort(myDict[i])
    
    for i in range(min_clus,len(myDict)): # map these indexes to the actual indexes in the interval.
        index_list=myDict[i]
        new_list=[]
        for j,elem in enumerate(index_list):
            new_list.append(int(map_index(elem)))
        myDict[i]=new_list

    dfs =[]
    for i in range(min_clus,len(myDict)):
        dfs.append(pd.DataFrame.from_dict(myDict[i]))

    if plot:
        fig, axs = plt.subplots(1,len(myDict), figsize=(15, 6), facecolor='w', edgecolor='k',sharex=True, sharey=True)
        fig.suptitle("Cluster assignment",fontsize=14) # change location
        plt.grid(b=None)


        norm = plt.Normalize(1, feature_df['Index'].values[-1])
        for i in range(min_clus,len(myDict)):
            col = plt.cm.Blues(norm(dfs[i].values))
            pd.DataFrame(data=np.repeat(1,len(norm(dfs[i].values)))).T.plot(kind='bar',stacked=True,ax=axs[i],width=1,legend=False,color=col)
            axs[i].set_xticks(ticks=[])
            axs[i].set_xlabel(f"{i+1}",rotation=90)
            axs[i].grid(False)

        fig.text(0.5, 0.04, 'Clusters', ha='center',fontsize=14)
        fig.text(0.07, 0.5, 'Number of samples', va='center', rotation='vertical',fontsize=14)


        sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
        sm.set_array([])
            
        cax = fig.add_axes([axs[-1].get_position().x1+0.01,axs[-1].get_position().y0,0.02,axs[-1].get_position().height])

        #plt.colorbar(sm,fraction=2.5, pad=1.5)
        cbar = plt.colorbar(sm,cax=cax)
        cbar.set_label('Interval index', rotation=270,labelpad=15)


        plt.show()

    clusterDict=myDict
    return clusterDict



def scale_df(df):
    df = np.asarray(df)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled

def find_optimal_dist(X):
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(X)
        distances, indices = nbrs.kneighbors(X)

        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        plt.plot(distances,color="#0F215A")
        plt.grid(True)
from sklearn.cluster import DBSCAN

def k_means_clustering(df,kind,knn_clusters=8,pca_components=10,plot=True):
    res = df
    cluster_df=res
    try:
        cluster_df= res.drop(labels=['Index'],axis=1) # Removing the index feature.
    except:
        print('Index does not exist')
    try:
        features_cleaned.drop(['cluster_assigned'],axis=1)
    except:
        # print('Col not there')
        pass
    scaled = scale_df(cluster_df)

    def dfScatter(df, xcol='Height', ycol='Weight', catcol='Gender'):
        fig, ax = plt.subplots(figsize=(15,4))
        # cmap=plt.get_cmap('tab20c')
        cmap = plt.get_cmap('Dark2')

        categories = np.unique(df[catcol])
        colors = np.linspace(0, 1, len(categories))
        colordict = dict(zip(categories, colors))  
        for i in range(len(colordict)):
            colordict[i+1] = cmap(colordict[i+1])
        df["Color"] = df[catcol].apply(lambda x: colordict[x])
        # df['Color'] = sns.color_palette("deep",len(colordict))
        ax.scatter(df[xcol], df[ycol], c=df.Color, s=40)
        axes = plt.gca()
        axes.yaxis.grid(True)
        axes.xaxis.grid(False)
        return fig,ax

    if type=='kmeans_pca':
        COMPONENTS = pca_components
        pca = PCA(n_components=COMPONENTS)
        principalComponents = pca.fit_transform(scaled)
        principalDf = pd.DataFrame(data = principalComponents
                 , columns = [f"{i}" for i in (range(COMPONENTS))])

        # Optimal number of clusters
        X = principalDf.values
    else:
        X = scaled

    wcss = []
    if plot:
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
    
    kmeans = KMeans(n_clusters=knn_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(X)

    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = res.index.values
    cluster = pred_y
    min_clus = cluster.min()
    if min_clus < 0:
        cluster = cluster + (min_clus*-1+1)    
    elif min_clus > 0:
        cluster = cluster - min_clus
    elif min_clus == 0:
        cluster = cluster + 1

    if plot:
        cluster_map['cluster'] = cluster
        fig,ax = dfScatter(cluster_map,xcol='data_index',ycol='cluster',catcol='cluster')
        plt.title("Clustering assignment for each time interval")
        plt.xlabel("Clusters")
        locs, labels = plt.yticks()            # Get locations and labels
        plt.yticks(np.arange(0, max(cluster)+1, 1.0))
        
        # Some figtext
        fig.text(0.04, -0.16, 'August\n2018', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontsize=14)
        fig.text(0.96, -0.16, 'January\n2020', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontsize=14)
        # ax.set_yticks(np.arange(locs.min(),locs.max()),np.arange(0,(cluster.max()+1)))
        plt.show()
    return pred_y
  
def db_scan_clustering(df,kind,eps=0.25,min_samples=5,pca_components=8,knn_clusters=8,plot=True):
    res = df
    cluster_df=res
    try:
        cluster_df= res.drop(labels=['Index'],axis=1) # Removing the index feature.
    except:
        print('Index does not exist')
    try:
        features_cleaned.drop(['cluster_assigned'],axis=1)
    except:
        pass
    scaled = scale_df(cluster_df)

    # Find the optimal epsilon


    def dfScatter(df, xcol='Height', ycol='Weight', catcol='Gender'):
        fig, ax = plt.subplots(figsize=(15,4))
        # cmap=plt.get_cmap('tab20c')
        cmap=plt.get_cmap('Dark2')

        ## sns.palplot(sns.color_palette("deep",20))


        categories = np.unique(df[catcol])
        colors = np.linspace(0, 1, len(categories))
        colordict = dict(zip(categories, colors))  
        for i in range(len(colordict)):
            colordict[i+1] = cmap(colordict[i+1])
        df["Color"] = df[catcol].apply(lambda x: colordict[x])
        # df['Color'] = sns.color_palette("deep",len(colordict))
        ax.scatter(df[xcol], df[ycol], c=df.Color, s=40)
        axes = plt.gca()
        axes.yaxis.grid(True)
        axes.xaxis.grid(False) 
        return fig,ax
    
    if kind=="raw":
        # Clustering with db scan
        find_optimal_dist(scaled)
        X = scaled
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        # print(clustering.labels_)
        cluster_map = pd.DataFrame()
        cluster_map['data_index'] = res.index.values
        cluster_map['real_index']=df['Index']
        cluster = clustering.labels_
        min_clus = cluster.min()
        if min_clus < 0:
            # print("Less than 0.")
            cluster = cluster + (min_clus*-1+1)    
            
        elif min_clus > 0:
            # print("More than 0.")
            cluster = cluster - min_clus
            
        elif min_clus == 0:
            # print("Equal 0.")
            cluster = cluster + 1

        cluster_map['cluster'] = cluster
        # cluster_map.plot.scatter(y='cluster', x='data_index',hue="cluster") # c="#0F215A"
        #sns.scatterplot(x=cluster_map['data_index'],y=cluster_map['cluster'],hue=cluster_map['cluster'])
        if plot:
            fig,ax = dfScatter(cluster_map,xcol='real_index',ycol='cluster',catcol='cluster')
            fig.suptitle("Clustering assignment for each time interval",fontsize=14)
            plt.xlabel(f"Interval number")
            #plt.grid(axis='y')
            ax.xaxis.grid(False)    

            locs, labels = plt.xticks()
            # Get locations and labels
            # plt.xticks(locs,labels)

            fig.text(0.04, -0.16, 'August\n2018', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontsize=14)
            fig.text(0.96, -0.16, 'January\n2020', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontsize=14)

            #plt.xticks(np.arange(0,(cluster_map['data_index'].values)[-1]),np.repeat)
            locs, labels = plt.yticks()            # Get locations and labels
            plt.yticks(np.arange(0, max(cluster)+1, 1.0))


            # ax.set_yticks(np.arange(locs.min(),locs.max()),np.arange(0,(cluster.max()+1)))

            plt.show()
            # plt.bar(cluster_map['cluster'].value_counts().keys(), cluster_map['cluster'].value_counts().values)
        return clustering.labels_,cluster_map

    if kind=="pca":
        from sklearn.decomposition import PCA
        COMPONENTS = pca_components # from the argument of the function call
        pca = PCA(n_components=COMPONENTS)
        principalComponents = pca.fit_transform(scaled)
        principalDf = pd.DataFrame(data = principalComponents
                     , columns = [f"{i}" for i in (range(COMPONENTS))])

        find_optimal_dist(principalDf.values) # optimal distance
        
        # clustring it
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(principalDf)
        cluster_map = pd.DataFrame()
        cluster_map['data_index'] = res.index.values
        cluster = clustering.labels_
        min_clus = cluster.min()
        if min_clus < 0:
            
            cluster = cluster + (min_clus*-1+1)    
        elif min_clus > 0:
            
            cluster = cluster - min_clus
        elif min_clus == 0:
            
            cluster = cluster + 1

        cluster_map['cluster'] = cluster
        if plot: 
            fig,ax = dfScatter(cluster_map,xcol='data_index',ycol='cluster',catcol='cluster')
            plt.title("Clustering assignment for each time interval")
            plt.xlabel("Clusters")
            plt.xticks(rotation=90)

            fig.text(0.04, -0.16, 'August\n2018', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontsize=14)
            fig.text(0.96, -0.16, 'January\n2020', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontsize=14)

            locs, labels = plt.yticks()            # Get locations and labels
            plt.yticks(np.arange(0, max(cluster)+1, 1.0))

            # ax.set_yticks(np.arange(locs.min(),locs.max()),np.arange(0,(cluster.max()+1)))

            plt.show()
        # plt.bar(cluster_map['cluster'].value_counts().keys(), cluster_map['cluster'].value_counts().values,color="#0F215A")
        # plt.xlabel('Cluster number')
        # plt.ylabel('Number of points in cluster')
        return clustering.labels_,principalDf


def plot_clusters_pair_plot(df, features, y_clusters):
    '''
    df:: The dataframe with all features for wt.
    features:: the featres in the pairplot to be studied. EX: ['ActPower','B1','B4']
    y_clusters:: The cluster assignment. 
    '''

    min_clus = y_clusters.min()
    if min_clus < 0:
        y_clusters = y_clusters + (min_clus*-1+1)    
    elif min_clus > 0:
        y_clusters = y_clusters - min_clus
    elif min_clus == 0:
        y_clusters = y_clusters + 1

    min_clus = 0
    df['cluster_assigned'] = y_clusters
    


    # hue='Index'
    ax=sns.pairplot(df,vars=features, hue='cluster_assigned',palette=sns.color_palette(sns.hls_palette(8, l=.4, s=.8), len(np.unique(y_clusters))))
    plt.show()
    #current_palette = sns.color_palette()
    #print(current_palette)
    #sns.palplot(current_palette)









