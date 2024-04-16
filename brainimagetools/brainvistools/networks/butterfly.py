import numpy as np
import matplotlib.pyplot as plt
    
def butterfly(Matrix = None,varargin = None): 
    #butterfly(Matrix, Quantile, Color, report)
    M = Matrix
    if len(varargin) > 2.9:
        C = varargin[2]
        Q = varargin[0]
    
    if len(varargin) == 2:
        Q = 0.95
        C = varargin[0]
    
    if len(varargin) == 1:
        Q = 0.95
        C = 0
    
    M_t = np.zeros((len(M),len(M)))
    cut = quantile(M,Q)
    for i in np.arange(1,len(M)+1).reshape(-1):
        for j in np.arange(1,len(M)+1).reshape(-1):
            if M(i,j) > cut:
                M_t[i,j] = M(i,j)
    
    G = graph(M_t,'upper')
    #Graph analysis
    eigenvector = centrality(G,'eigenvector')
    pagerank = centrality(G,'pagerank')
    betweenness = centrality(G,'betweenness')
    closeness = centrality(G,'closeness')
    deg = centrality(G,'degree')
    path = distances(G)
    mpath = mean(path)
    mpath = np.transpose(mpath)
    charpath = mean(mpath)
    efficiency = 1.0 / charpath
    if C == 0:
        C = deg
    else:
        if C(2) == 'i':
            C = eigenvector
        else:
            if C(1) == 'p':
                C = pagerank
            else:
                if C(1) == 'b':
                    C = betweenness
                else:
                    if C(1) == 'c':
                        C = closeness
                    else:
                        if C(1) == 'd':
                            C = deg
                        else:
                            if C(1) == 'e':
                                C = 1.0 / mpath
                            else:
                                C = deg
    
    plt.plot(G,'MarkerSize',0.1 + deg / (0.2 * mean(deg) + 1),'NodeCData',C,'EdgeAlpha',0.15,'EdgeColor','k','Layout','force')
    if len(varargin) == 4:
        print('The graph has the following characteristics:')
        print('')
        print('Centrality measures')
        print('Mean Eigenvector centrality = ' + string(mean(eigenvector)))
        print('Mean pagerank centrality = ' + string(mean(pagerank)))
        print('Mean betweenness centrality = ' + string(mean(betweenness)))
        print('Mean closeness centrality = ' + string(mean(closeness)))
        print('Mean degree = ' + string(mean(deg)))
        print('')
        print('Connectivity measures')
        print('')
        print('Mean Connectivity = ' + string(mean(mean(M))))
        print('Characteristic path length = ' + string(charpath))
        print('Global efficiency = ' + string(efficiency))
    
    return eigenvector,pagerank,betweenness,closeness,deg,mpath
    
    return eigenvector,pagerank,betweenness,closeness,deg,mpath