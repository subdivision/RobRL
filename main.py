import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta

NUM_OF_ITERS = 5000

#-----------------------------------------------------------------------------
def init():
    edges = [(0,4),(4,0),(0,3),(3,0),(1,2),(2,1),(1,4),(4,1),
             (1,8),(8,1),(1,9),(9,1),(2,3),(3,2),(2,6),(6,2),
             (1,5),(5,1),(2,5),(5,2),(5,6),(6,5),(7,8),(8,7),
             (7,5),(5,7),(8,9),(9,8),(8,10),(10,8),(9,10),(10,9)]
    G = nx.Graph()
    G.add_edges_from(edges)
    # pos = nx.spring_layout(G)
    # nx.draw_networkx_nodes(G, pos)
    # nx.draw_networkx_edges(G, pos)
    # nx.draw_networkx_labels(G, pos)
    # plt.show()
    Reward = np.matrix(np.zeros(shape=(11,11)))
    for x in G[10]:
        Reward[x,10] = 100

    QVals = np.matrix(np.zeros(shape=(11,11)))
    QVals -= 100
    for curr in G.nodes:
        for x in G[curr]:
            QVals[x,curr] = 0
            QVals[curr,x] = 0
    return G, Reward, QVals

#-----------------------------------------------------------------------------
def next_stop(start, err, G, QVals):
    random_val = random.uniform(0,1)
    if random_val < err:
        sample = G[start]
    else:
        sample = np.where( QVals[start] == np.max(QVals[start,]) )[1]
    next_node = int(np.random.choice(sample,1))
    return next_node

#-----------------------------------------------------------------------------
def update_QVals(node1, node2, learning_rate, discount, Reward, QVals):
    max_index = np.where(QVals[node2] == np.max(QVals[node2,]))[1]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size = 1))
    else:
        max_indexs = int(max_index)
    max_value = QVals[node2, max_index]
    QVals[node1, node2] = int(   (1.-learning_rate) * QVals[node1,node2] \
                               +     learning_rate  * ( Reward[node1, node2] \
                                                        + discount * max_value ) )

#-----------------------------------------------------------------------------
def learn(err, learning_rate, discount, G, Reward, QVals):
    for i in range(NUM_OF_ITERS):
        curr_node = random.randint(0, 10)
        next_node = next_stop( curr_node, err, G, QVals )
        update_QVals( curr_node, next_node, learning_rate, discount, Reward, QVals )

#-----------------------------------------------------------------------------
def extract_path(begin, end, QVals):
    path= [begin]
    next_node = np.argmax(QVals[begin,])
    path.append(next_node)
    while next_node != end:
        next_node = np.argmax(QVals[next_node,])
        path.append(next_node)
    return path

#-----------------------------------------------------------------------------
def main():
    strt = datetime.now()
    G, Reward, QVals = init()
    learn(0.5, 0.8, 0.8, G, Reward, QVals)
    result = extract_path(0, 10, QVals)
    fnsh = datetime.now()
    print( result )
    print(str( fnsh-strt ))

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    main()

#================================= END OF FILE ===============================