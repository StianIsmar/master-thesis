from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pandas as pd

def plot_clusters(y_cluster_kmeans):
    '''
    y_cluster_kmeans :: List of all datapoints and which cluster they are assigned to.
    
    EXAMPLE:
    plot_clusters([2, 2, 1, 1, 1, 2, 3]). Datapoint at index 0 is assigned to cluster 2, index 1 has the same cluster etc.
    
    '''    
    myDict = {}
    min_clus = y_cluster_kmeans.min()
    
    if not (min_clus == 0):
        if min_clus < 0:
            y_cluster_kmeans = y_cluster_kmeans + (min_clus*-1 )    
        elif min_clus > 0:
            y_cluster_kmeans = y_cluster_kmeans - min_clus
        min_clus = 0
    
    for i, elem in enumerate(y_cluster_kmeans):
        if (elem in myDict):
            myDict[elem].append(i)
        else:
            myDict[elem] = [i]
    # sorted plot matplotlib

    for i in range(min_clus,len(myDict)):

        np.sort(myDict[i])
    dfs =[]
    for i in range(min_clus,len(myDict)):
        dfs.append(pd.DataFrame.from_dict(myDict[i]))

    fig, axs = plt.subplots(1,len(myDict), figsize=(15, 6), facecolor='w', edgecolor='k',sharex=True, sharey=True)
    fig.subplots_adjust(hspace = .2, wspace=1)

    # ax.set_xticks(np.arange(len(myDict)))

    norm = plt.Normalize(0, len(y_cluster_kmeans))
    # col = plt.cm.Blues(norm(dfs[0].values))
    #dfs[0].T.plot(kind='bar',stacked=True,width=0.5,legend=False, color=col,ax=ax)
    for i in range(min_clus,len(myDict)):
        col = plt.cm.Blues(norm(dfs[i].values))
        pd.DataFrame(data=np.repeat(1,len(norm(dfs[i].values)))).T.plot(kind='bar',stacked=True,ax=axs[i],width=1,legend=False,color=col)
        axs[i].set_xticks(ticks=[])
        axs[i].set_xlabel(f"{i+1}",rotation=90)

    fig.text(0.5, 0.04, 'Clusters', ha='center')
    fig.text(0.07, 0.5, 'Number of samples', va='center', rotation='vertical')

    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])
        
    cax = fig.add_axes([axs[-1].get_position().x1+0.01,axs[-1].get_position().y0,0.02,axs[-1].get_position().height])

    #plt.colorbar(sm,fraction=2.5, pad=1.5)
    cbar = plt.colorbar(sm,cax=cax)
    cbar.set_label('Interval index', rotation=270,labelpad=15)
    plt.title("Cluster assignment") # change location


    plt.show()



def scale_df(df):
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

def db_scan_clustering(df,kind,eps=0.25,pca_components=8):
    res = df
    cluster_df= res.drop(labels=['Index','NacelleDirection'],axis=1)
    scaled = scale_df(cluster_df)

    # Find the optimal epsilon
    import numpy as np
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import DBSCAN
    from matplotlib import pyplot as plt
    import seaborn as sns
    from matplotlib import cm
    import pandas as pd


    def dfScatter(df, xcol='Height', ycol='Weight', catcol='Gender'):
        fig, ax = plt.subplots(figsize=(15,6))
        cmap=plt.get_cmap('tab20c')

        ## sns.palplot(sns.color_palette("deep",20))


        categories = np.unique(df[catcol])
        colors = np.linspace(0, 1, len(categories))
        colordict = dict(zip(categories, colors))  
        for i in range(len(colordict)):
            colordict[i+1] = cmap(colordict[i+1])
        df["Color"] = df[catcol].apply(lambda x: colordict[x])
        # df['Color'] = sns.color_palette("deep",len(colordict))
        ax.scatter(df[xcol], df[ycol], c=df.Color)
        axes = plt.gca()
        axes.yaxis.grid()
        return fig,ax
        

    
    if kind=="raw":
        # Clustering with db scan
        find_optimal_dist(scaled)
        X = scaled
        clustering = DBSCAN(eps=0.25, min_samples=2).fit(X)
        cluster_map = pd.DataFrame()
        cluster_map['data_index'] = res.index.values
        cluster = clustering.labels_
        min_clus = cluster.min()
        if min_clus < 0:
            print("Less than 0.")
            cluster = cluster + (min_clus*-1+1)    
        elif min_clus > 0:
            print("More than 0.")
            cluster = cluster - min_clus
        elif min_clus == 0:
            print("Equal 0.")
            cluster = cluster + 1

        cluster_map['cluster'] = cluster
        # cluster_map.plot.scatter(y='cluster', x='data_index',hue="cluster") # c="#0F215A"
        #sns.scatterplot(x=cluster_map['data_index'],y=cluster_map['cluster'],hue=cluster_map['cluster'])
        fig,ax = dfScatter(cluster_map,xcol='data_index',ycol='cluster',catcol='cluster')
        plt.title("Clustering assignment over time")
        plt.xlabel("Clusters")
        locs, labels = plt.yticks()            # Get locations and labels
        plt.yticks(np.arange(0, max(cluster)+1, 1.0))
        plt.margins(0)

        # ax.set_yticks(np.arange(locs.min(),locs.max()),np.arange(0,(cluster.max()+1)))

        plt.show()
        # plt.bar(cluster_map['cluster'].value_counts().keys(), cluster_map['cluster'].value_counts().values)
        return clustering.labels_

    if kind=="pca":
        from sklearn.decomposition import PCA
        COMPONENTS = pca_components # from the argument of the function call
        pca = PCA(n_components=COMPONENTS)
        principalComponents = pca.fit_transform(scaled)
        principalDf = pd.DataFrame(data = principalComponents
                     , columns = [f"{i}" for i in (range(COMPONENTS))])

        find_optimal_dist(principalDf.values) # optimal distance
        
        # clustring it
        clustering = DBSCAN(eps=eps, min_samples=5).fit(principalDf)
        cluster_map = pd.DataFrame()
        cluster_map['data_index'] = res.index.values
        cluster = clustering.labels_
        min_clus = cluster.min()
        if min_clus < 0:
            print("Less than 0.")
            cluster = cluster + (min_clus*-1+1)    
        elif min_clus > 0:
            print("More than 0.")
            cluster = cluster - min_clus
        elif min_clus == 0:
            print("Equal 0.")
            cluster = cluster + 1

        cluster_map['cluster'] = cluster
        fig,ax = dfScatter(cluster_map,xcol='data_index',ycol='cluster',catcol='cluster')
        plt.title("Clustering assignment over time")
        plt.xlabel("Clusters")
        locs, labels = plt.yticks()            # Get locations and labels
        plt.yticks(np.arange(0, max(cluster)+1, 1.0))
        plt.margins(0)

        # ax.set_yticks(np.arange(locs.min(),locs.max()),np.arange(0,(cluster.max()+1)))

        plt.show()
        # plt.bar(cluster_map['cluster'].value_counts().keys(), cluster_map['cluster'].value_counts().values,color="#0F215A")
        # plt.xlabel('Cluster number')
        # plt.ylabel('Number of points in cluster')
        return clustering.labels_,principalDf

        '''
        if kind=="pca":
        from sklearn.decomposition import PCA
        COMPONENTS = pca_components # from the argument of the function call
        pca = PCA(n_components=COMPONENTS)
        principalComponents = pca.fit_transform(scaled)
        principalDf = pd.DataFrame(data = principalComponents
                     , columns = [f"{i}" for i in (range(COMPONENTS))])

        find_optimal_dist(principalDf.values) # optimal distance
        
        # clustring it
        clustering = DBSCAN(eps=eps, min_samples=5).fit(principalDf)
        cluster_map = pd.DataFrame()
        cluster_map['data_index'] = res.index.values
        cluster_map['cluster'] = clustering.labels_
        cluster_map.plot.scatter(y='cluster', x='data_index',c="#0F215A")
        plt.xlabel(f'Data point Timestamp [0 ->{res.index.values.max()}]')
        plt.ylabel('Cluster')
        plt.show()
        # plt.bar(cluster_map['cluster'].value_counts().keys(), cluster_map['cluster'].value_counts().values,color="#0F215A")
        # plt.xlabel('Cluster number')
        # plt.ylabel('Number of points in cluster')
        return clustering.labels_,principalDf
        '''



    if kind=="knn":
        import numpy as np
        import pandas as pd
        from matplotlib import pyplot as plt
        from sklearn.datasets.samples_generator import make_blobs
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        COMPONENTS = 8
        pca = PCA(n_components=COMPONENTS)
        principalComponents = pca.fit_transform(scaled)
        principalDf = pd.DataFrame(data = principalComponents
                     , columns = [f"{i}" for i in (range(COMPONENTS))])


        # Optimal distance
        X = principalDf.values

        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()


        
        kmeans = KMeans(n_clusters=8, init='k-means++', max_iter=300, n_init=10, random_state=0)
        pred_y = kmeans.fit_predict(X)
        print((pred_y)) # the clusters assigned

   
        return pred_y


def plot_clusters_pair_plot(df, features, y_clusters):
    '''
    df:: The dataframe with all features for wt.
    features:: the featres in the pairplot to be studied. EX: ['ActPower','B1','B4']
    y_clusters:: The cluster assignment. 
    '''
    print("THIS IS THE MIN:", y_clusters.min())

    min_clus = y_clusters.min()
    if min_clus < 0:
        y_clusters = y_clusters + (min_clus*-1+1)    
    elif min_clus > 0:
        y_clusters = y_clusters - min_clus
    elif min_clus == 0:
        y_clusters = y_clusters + 1

    min_clus = 0
    df['cluster_assigned'] = y_clusters
    print("THIS IS THE MIN:", y_clusters.min())


    # hue='Index'
    ax=sns.pairplot(df,vars=features, hue='cluster_assigned')
    plt.show()
    #current_palette = sns.color_palette()
    #print(current_palette)
    #sns.palplot(current_palette)









