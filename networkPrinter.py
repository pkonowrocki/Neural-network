import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def initialize():
    plt.ion()

def print_regression(net, X, Y, size = 5, dx = 0.1):
    plt.figure(15)
    plt.ion()
    plt.clf()
    plt.plot(np.reshape(X, -1), np.reshape(Y, -1), marker='.', linestyle = 'None')
    x = np.arange(-size, size, dx)
    y = []
    for i in x:
        j = net.forward(np.array([[i]]))
        y.append(j)
    plt.plot(np.reshape(x, -1),np.reshape(y, -1))
    plt.show()

def _print_points(X,Y):
    if max(Y[0].shape)==3:
        cmap_bold  = ['#FF0000', '#00FF00', '#0000FF']
        X1 = []
        X2 = []
        X3 = []
        Y1 =[]
        Y2 =[]
        Y3 =[]
        for i in range(len(Y)):
            if np.argmax(Y[i])==0:
                X1.append(X[i][0])
                Y1.append(X[i][1])
            elif np.argmax(Y[i])==1:
                X2.append(X[i][0])
                Y2.append(X[i][1])
            else:
                X3.append(X[i][0])
                Y3.append(X[i][1])
        plt.plot(X1, Y1, c=cmap_bold[0], marker='.', linestyle = 'None')
        plt.plot(X2, Y2, c=cmap_bold[1], marker='.', linestyle = 'None')
        plt.plot(X3, Y3, c=cmap_bold[2], marker='.', linestyle = 'None')
        plt.draw()
    else:
        cmap_bold  = ['#FF0000', '#00FF00']
        X1 = []
        Y1 = []
        X2 = []
        Y2 = []
        for i in range(len(Y)):
            if Y[i]==0:
                X1.append(X[i][0])
                Y1.append(X[i][1])
            else:
                X2.append(X[i][0])
                Y2.append(X[i][1])
        plt.plot(X1, Y1, c=cmap_bold[0], marker='.', linestyle = 'None')
        plt.plot(X2, Y2, c=cmap_bold[1], marker='.', linestyle = 'None')
        plt.draw()

def print_classification(net, X, Y, size = 1.5, dS = 0.1):
    cmap_light = []
    x_min = -size
    x_max = size
    y_min = -size
    y_max = size
    dx = dS
    dy = dS
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, dx), np.arange(y_min, y_max, dy))
    z = np.zeros(xx.shape)
    plt.figure(10)
    plt.clf()
    _print_points(X,Y)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            inp = np.array([[xx[i][j], yy[i][j]]]).T
            ret = net.forward(inp)
            if(np.max(ret.shape)==3):
                cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
                if np.argmax(ret)==0:
                    ret=0
                elif np.argmax(ret)==1:
                    ret=1
                else:
                    ret=2
                z[i][j] = ret
            else:
                cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
                z[i][j] = ret
    
    plt.pcolormesh(xx, yy, z, cmap=cmap_light)
    plt.draw()
    plt.show()

def print_network(net, print_text = False):
    G = nx.DiGraph()
    pos = {}
    for i in range(1, net.numberOfLayers+1):
        W = net.paramsValues['W' + str(i)]
        
        if i == 1:
            if (print_text):
                print(f'Layer 0')
            print_layer(G, pos, net, 0)
        
        if (print_text):
            print(f'Layer {i}')
        print_layer(G, pos, net, i)
        print_edges(G, net, W, i-1, i)

        if (print_text):
            print()

    # nodes style
    nx.draw_networkx_nodes(G, pos, node_size=500)
    # edges style
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    arc_weight=nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=arc_weight)
    plt.figure(69)
    plt.draw()
    plt.show()

def print_layer(G, pos, net, i, print_text = False):
    for k in range(1, net.layerSize[i] + 1):
        if print_text:
            print(f'adding node L{i}N{k}')
        node = f'L{i}N{k}'
        G.add_node(node)
        m = (net.biggestLayerSize - net.layerSize[i]) / 2
        pos[node] = (i, k + m)

def print_edges(G, net, W, a, b):
    for j in range(net.layerSize[b]):
            for k in range(net.layerSize[a]):
                G.add_edge(f'L{a}N{k+1}', f'L{b}N{j+1}', weight=np.round(W[j][k],3))

def print_error(error):
    plt.figure(1)
    plt.ion()
    plt.clf()
    plt.plot(error)
    plt.xlabel("Epochs", fontsize = 10)
    plt.ylabel("Mean error value", fontsize = 10)
    plt.legend()
    plt.show()

def print_error(errorTraining, errorValidate):
    plt.figure(1)
    plt.ion()
    plt.clf()
    plt.plot(errorTraining, label='training error')
    plt.plot(errorValidate, label='validation error')
    plt.title(f'Training error: {errorTraining[::-1][0]}, validation error: {errorValidate[::-1][0]}')
    plt.xlabel("Epochs", fontsize = 10)
    plt.ylabel("Mean error value", fontsize = 10)
    plt.legend()
    plt.show()

def print_error_hist(error):
    plt.figure(2)
    plt.clf()
    plt.hist(error)
    plt.show()

def wait(seconds = 0.01):
    plt.pause(seconds)