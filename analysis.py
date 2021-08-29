# Anthony Zheng

import numpy
import random
import copy
from collections import deque
import matplotlib.pyplot as plt
def compute_largest_cc_size(g: dict) -> int:
    """
    calculate the size of the largest connected component in
    graph g

    Arguments:
    g -- dictionary of nodes and their adjacency lists

    Returns:
    the integer size of nodes in the largest connected component in g
    """
    maxconnected = 0
    #initialize a queue
    q = deque()
    #make a dictionary mapping each node to whether or not it has been visited
    keys = list(g.keys())
    nodes = {keys[i]: False for i in range(len(g))}
    #for every node in the graph
    for node in keys:
        #if it has been explored, skip it
        if nodes.get(node):
            continue
        #if not, add to queue and set explored to true
        q.appendleft(node)
        nodes[node] = True
        con = 0
        #while the queue is not empty
        while q:
            #remove the element at the front of the queue
            j = q.pop()
            #add one to the connected component size
            con += 1
            if g.get(j):
                #for all neigbors of the node
                for neighbor in g.get(j):
                    #if the neighbor has not been visited
                    if not nodes[neighbor]:
                        #add the neighbor to the queue and set it to visited
                        q.appendleft(neighbor)
                        nodes[neighbor] = True
        #if this connected component is larger than the current max, set its value to the max
        if con > maxconnected:
            maxconnected = con
    return maxconnected

def upa(n, m):
    """
    Generate an undirected graph with n node and m edges per node
    using the preferential attachment algorithm.

    Arguments:
    n -- number of nodes
    m -- number of edges per node

    Returns:
    undirected random graph in UPAG(n, m)
    """
    g = {}
    if m <= n:
        g = make_complete_graph(m)
        for new_node in range(m, n):
            # Find <=m nodes to attach to new_node
            totdeg = float(total_degree(g))
            nodes = list(g.keys())
            probs = []
            for node in nodes:
                probs.append(len(g[node]) / totdeg)
            mult = distinct_multinomial(m, probs)

            # Add new_node and its random neighbors
            g[new_node] = set()
            for idx in mult:
                node = nodes[idx]
                g[new_node].add(node)
                g[node].add(new_node)
    return g

def erdos_renyi(n, p):
    """
    Generate a random Erdos-Renyi graph with n nodes and edge probability p.

    Arguments:
    n -- number of nodes
    p -- probability of an edge between any pair of nodes

    Returns:
    undirected random graph in G(n, p)
    """
    g = {}

    ### Add n nodes to the graph
    for node in range(n):
        g[node] = set()

    ### Iterate through each possible edge and add it with
    ### probability p.
    for u in range(n):
        for v in range(u+1, n):
            r = random.random()
            if r < p:
                g[u].add(v)
                g[v].add(u)

    return g

def total_degree(g):
    """
    Compute total degree of the undirected graph g.

    Arguments:
    g -- undirected graph

    Returns:
    Total degree of all nodes in g
    """
    return sum(map(len, g.values()))

def make_complete_graph(num_nodes):
    """
    Returns a complete graph containing num_nodes nodes.

    The nodes of the returned graph will be 0...(num_nodes-1) if num_nodes-1 is positive.
    An empty graph will be returned in all other cases.

    Arguments:
    num_nodes -- The number of nodes in the returned graph.

    Returns:
    A complete graph in dictionary form.
    """
    result = {}

    for node_key in range(num_nodes):
        result[node_key] = set()
        for node_value in range(num_nodes):
            if node_key != node_value:
                result[node_key].add(node_value)

    return result

def distinct_multinomial(ntrials, probs):
    """
    Draw ntrials samples from a multinomial distribution given by
    probs.  Return a list of indices into probs for all distinct
    elements that were selected.  Always returns a list with between 1
    and ntrials elements.

    Arguments:
    ntrials -- number of trials
    probs   -- probability vector for the multinomial, must sum to 1

    Returns:
    A list of indices into probs for each element that was chosen one
    or more times.  If an element was chosen more than once, it will
    only appear once in the result.
    """
    ### select ntrials elements randomly
    mult = numpy.random.multinomial(ntrials, probs)

    ### turn the results into a list of indices without duplicates
    result = [i for i, v in enumerate(mult) if v > 0]
    return result

def read_graph(filename):
    """
    Read a graph from a file.  The file is assumed to hold a graph
    that was written via the write_graph function.

    Arguments:
    filename -- name of file that contains the graph

    Returns:
    The graph that was stored in the input file.
    """
    with open(filename) as f:
        g = eval(f.read())
    return g

def _dict2lists(data):
    """
    Convert a dictionary into a list of keys and values, sorted by
    key.

    Arguments:
    data -- dictionary

    Returns:
    A tuple of two lists: the first is the keys, the second is the values
    """
    xvals = list(data.keys())
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals

def copy_graph(g):
    """
    Return a copy of the input graph, g

    Arguments:
    g -- a graph

    Returns:
    A copy of the input graph that does not share any objects.
    """
    return copy.deepcopy(g)

def randomAttack(g):
    """
    remove 20% of the random nodes of graph g and store the connected component
    sizes every time one is removed

    Arguments:
    g -- dictionary of nodes and their adjacency lists

    Returns:
    a list of connected component sizes
    """
    #we need to store largest connected component data
    CC = []
    # number of nodes = length of graph dictionary
    nodes = len(g)
    #we are only interested in removing 20%
    while len(CC) < 0.2*nodes:
        #split the dictionary g into node and neighbor list
        x, y = _dict2lists(g)
        #choose a random node to remove
        num = random.choice(x)
        #remove that node
        g.pop(num)
        #remove the chosen node from all neighbor sets
        g = removenode(g, num)
        #add the connected component size to data list
        CC.append(compute_largest_cc_size(g))
    return CC

def targetedAttack(g):
    """
    remove 20% of the nodes of g, choosing the node
    with the highest degree to remove, and store the largest
    connected componennt size each time a node is removed

    Arguments:
    g -- dictionary of nodes and their adjacency lists

    Returns:
    list of connected component sizes
    """
    #we need to store the connected component data
    CC = []
    #the number of nodes = the length of the g dictionary
    nodes = len(g)
    #we are only interested in the first 20% removed
    while len(CC) < 0.2*nodes:
        #split the graph dictionary into nodes and neighbors
        x, y = _dict2lists(g)
        #max index and length of neighbor set, set to 0 first
        maxind = 0
        maxlen = 0
        #for each neighbor set
        for set in y:
            #if the set length is greater than the stored max
            if set and len(set) > maxlen:
                #set max length and index to this set
                maxlen = len(set)
                maxind = y.index(set)
        #remove the highest degree node
        g.pop(x[maxind])
        #remove that node from the neighbor sets
        g = removenode(g, x[maxind])
        #add the conencted componenet size to the data list
        CC.append(compute_largest_cc_size(g))
    return CC

def removenode(g, removed):
    """
    removes the specified node from all of the adjacency lists it
    is in

    Arguments:
    g -- dictionary of nodes and their adjacency lists
    removed -- the name of the node to be removed

    Returns:
    modified g, with specified node removed from all adjacency lists
    """
    #for every neighbor
    for neighbor in g:
        #if removed is in the adjacency list
        if g[neighbor] and removed in g[neighbor]:
            #remove it from the list
            g[neighbor].discard(removed)
    return g

#read the specified graph
providedGraphRand = read_graph("rf7.repr")
#calculate the number of nodes
nodes = len(providedGraphRand)
#calculate the total degree of the graph
totdeg = total_degree(providedGraphRand)
#generate an erdos renyi graph and UPA graph
#with similar edge characteristics to the provided
ERrand = erdos_renyi(nodes, totdeg/(nodes*(nodes-1)))
UPArand = upa(nodes, round(totdeg/(nodes*2)))
providedGraphTarg = read_graph("rf7.repr")

#copy both randomly generated graphs
ERtarg = copy_graph(ERrand)
UPAtarg = copy_graph(UPArand)

#run targeted ad random attacks on the generated graphs, store the returned data
CCERrand =  randomAttack(ERrand)
CCERtarg = targetedAttack(ERtarg)

CCUPArand = randomAttack(UPArand)
CCUPAtarg = targetedAttack(UPAtarg)

#run targeted and random attacks on the provided graph, store the returned data
CCprovtarg = targetedAttack(providedGraphTarg)
CCprovrand = randomAttack(providedGraphRand)

#graphing
xaxis = numpy.arange(0,len(CCprovrand))
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
ax.plot(xaxis, CCERrand, label = 'erdos_renyi random attack', color = 'r')
ax.plot(xaxis, CCERtarg, label = 'erdos_renyi targeted attack', color = 'g')
ax.plot(xaxis, CCUPArand, label = 'UPA random attack', color = 'b')
ax.plot(xaxis, CCUPAtarg, label = 'UPA targeted attack', color = 'y')
ax.plot(xaxis, CCprovrand, label = 'provided random attack', color = 'm')
ax.plot(xaxis, CCprovtarg, label = 'provided targeted attack', color = 'c')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 6})
ax.set_xlabel("number of removed nodes")
ax.set_ylabel("largest connected component")
plt.title("network resilience")
plt.grid(True)
plt.show()

