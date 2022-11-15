import statistics
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx
from scipy.signal import find_peaks

class Gphase:
    """
    Create the phase diagram for a weighted graph.

    Args:
        degree (list): the list of degree
        weight (list): the list of weight
    """

    def __init__(
        self,
        degree: list,
        weight: list,
        max_lam: Optional[str] = 4
    ):

        self.degree = degree
        self.weight = weight
        self.max_lam = max_lam

    #entropy from a list
    def entropy(self, l):
        l_copy = l.copy()
        l_copy.sort()
        l_diff = np.diff(l_copy)

        e = 0
        for i in l_diff:
            if i==0:
                pass
            else:
                e += -i*np.log2(i)
    
        return e

    #calculate phase diagram
    def var_phase(self, mode, max_lam: Optional[int]= None):
        """
        This calculates the phase (normalized variance) for the weighted diagram. 
        Args:
            max_lam: int that is the upper bound of the parameter lambda.
        Return:
            lambda list
            variance list
        """
        if max_lam == None:
            max_lam = self.max_lam
        d = self.degree
        w = self.weight

        #shift and normalize (against range)
        w = [(i-min(w))/(max(w)-min(w)) for i in w]

        d = [(i-min(d))/(max(d)-min(d)) for i in d]

        lam_list = [i/1000 for i in range(0,max_lam*1000)]

        p_list = []

        for lam in lam_list:
            p_list.append(np.array([i-lam*j for i,j in zip(d,w)]))

        #shift and normalize
        p_list = [(p_list[i]-min(p_list[i]))/(max(p_list[i])-min(p_list[i])) for i in range(len(p_list))]

        #shift and normalize
        p_list_sn = [(p_list[i]-min(p_list[i]))/(max(p_list[i])-min(p_list[i])) for i in range(len(p_list))]

        #shit s.t. average at 0
        p_list = [-sum(p_list[i])/len(p_list[i])+p_list[i] for i in range(len(p_list))]
        
        #chi_list = [statistics.variance(p_list[i]) for i in range(len(p_list))]
        if mode == 'entropy':
            chi_list = [self.entropy(p_list_sn[i]) for i in range(len(p_list_sn))]
        elif mode == 'repeat':
            chi_list = [ len(p_list_sn[i])-len(set(np.around(p_list_sn[i], decimals=3))) for i in range(len(p_list))]
        else:
            raise Exception('no such mode')

        return lam_list, chi_list

    def reduced_w(self, d, w, lam):
        """
        This method calcuates the reduced weight for a given lambda.
        Args:
            d: list of degree
            w: list of weight
            lam: [float] chosen lambda
        Return:
            reduced weight list
        """

        #shift and normalize (against range)
        w = [(i-min(w))/(max(w)-min(w)) for i in w]

        d = [(i-min(d))/(max(d)-min(d)) for i in d]

        p_list = []

        p_list = np.array([i-lam*j for i,j in zip(d,w)])

        #shift and normalize
        p_list = (p_list-min(p_list))/(max(p_list)-min(p_list))

        #shit s.t. average at 0
        p_list = -sum(p_list)/len(p_list)+p_list
        
        return p_list

    #simple connected graph generator
    def random_graph_gen(self):
        """
        This method generates the graph object.
        Return:
            graph obj
        """

        degree = self.degree

        G = nx.random_degree_sequence_graph(degree) 
        while not nx.is_connected(G):
            G = nx.random_degree_sequence_graph(degree)

        return G

    def verify_list(self, max_ind, lambda_list):
        """
        This method generates the corresponding reduced weight list and chosen lambdas.
        Args:
            max_ind: list of transition points.
            lambda_list: list of lambda range
        Return:
            list of chosen lambda
            list of reduced weight corresponding to the chosen lambdas
        """
        lam = []
        max_ind = np.append(max_ind, max(lambda_list))
        max_ind = np.append(np.array([0]), max_ind)
        add_list = np.diff(max_ind)/2

        for i in range(len(add_list)):
            lam.append(max_ind[i]+add_list[i])

        weight_list = []
        for i in lam:
            weight_list.append(self.reduced_w(self.degree, self.weight, i))

        return lam, weight_list

    #redefined remove list element, used for next function
    @classmethod
    def _remove(self, l, element):
        if element in l:
            l.remove(element)
        else:
            pass

    #this excutes the sequencial algorithm and calculate the total weight, output the MWIS and weight
    def total_weight(self, graph,reduced_w,w):
        """
        This method calcuates the total weight, given graph, reduced weight list, original weight list
        Args:
            graph: [graph] graph object
            reduced_w: list of reduced weight list
            w: list of original weight
        Return:
            list of nodes in the solution set
            maximum weight
        """
        I = []
        V_prime = list(range(nx.number_of_nodes(graph)))
        reduced_w_copy = reduced_w.copy()
        max_w = max(reduced_w_copy)+1

        while len(V_prime) != 0:
            i_min = np.argmin(reduced_w_copy)
            I.append(i_min)
            self._remove(V_prime, i_min)
            reduced_w_copy[i_min] = max_w
            for n in list(nx.all_neighbors(graph, i_min)):
                self._remove(V_prime, n)
                reduced_w_copy[n] = max_w

        w_tot = 0
        for i in I:
            w_tot += w[i]

        return I, w_tot

    #find transition points
    def transition(self, chi_list):
        """
        This method calculates the 1st and 2nd order derivative. And get the transition points.
        Args:
            chi_list: list of variance
        Return:
            list of transition points
        """

        m_index = np.array(find_peaks(chi_list, prominence=0.1)[0])/1000

        return m_index


    #calcuate the 1st, 2nd order derivative and the transition point
    def dev(self, lam_list, chi_list):
        """
        This method calculates the 1st and 2nd order derivative. And get the transition points.
        Args:
            lam_list: list of lambda range
            chi_list: list of variance
        Return:
            list of 1st order derivative
            list of 2nd order derivative
            list of transition points
        """

        dev1 = np.diff(chi_list)/np.diff(lam_list)
        dev2 = np.diff(dev1)/np.diff(lam_list[:-1])/10
        #m_index = np.array(argrelextrema(dev2, np.greater))[0]/100
        m_index = np.array(find_peaks(dev2, prominence=10, threshold=10)[0])/1000

        return dev1, dev2, m_index

    #you can also calculate phases explicitly for small graph
    def phase_all(self, G, lam_list):
        """
        Plot phases explicitly, for small graph.
        Args:
            G: graph
            lam_list: lambda list
        """
        w_list = []
        #I_list = []

        for l in lam_list:
            
            w_re = self.reduced_w(self.degree, self.weight, l)
            I, w_tot = self.total_weight(G, w_re, self.weight)
            #I_list.append(I)
            w_list.append(w_tot)

        return w_list
