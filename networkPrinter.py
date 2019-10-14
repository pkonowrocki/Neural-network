import networkx as nx
import matplotlib.pyplot as plt

def initialize():
    # this causes printing to be asynchronious
    plt.ion()

def print_network(net):
    print('start')
    G = nx.DiGraph()
    pos = {}
    for i in range(1, net.numberOfLayers+1):
        W = net.paramsValues['W' + str(i)]
        if i == 1:
            print(f'Layer 0')
            print_layer(G, pos, net, 0)

        print(f'Layer {i}')
        print_layer(G, pos, net, i)
        print_edges(G, net, W, i-1, i)

        print()

    # nodes style
    nx.draw_networkx_nodes(G, pos, node_size=500)
    # edges style
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    arc_weight=nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=arc_weight)
    plt.draw()
    plt.show()
    print('end')

def print_layer(G, pos, net, i):
    for k in range(1, net.layerSize[i] + 1):
        print(f'adding node L{i}N{k}')
        node = f'L{i}N{k}'
        G.add_node(node)
        m = (net.biggestLayerSize - net.layerSize[i]) / 2
        pos[node] = (i, k + m)

def print_edges(G, net, W, a, b):
    for j in range(net.layerSize[b]):
            for k in range(net.layerSize[a]):
                G.add_edge(f'L{a}N{k+1}', f'L{b}N{j+1}', weight=W[j][k])