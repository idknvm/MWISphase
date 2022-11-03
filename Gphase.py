import statistics
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx
from scipy.signal import argrelextrema

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

    #calculate phase diagram
    def var_phase(self, max_lam: Optional[int]= None):
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

        lam_list = [i/100 for i in range(0,max_lam*100)]

        p_list = []

        for lam in lam_list:
            p_list.append(np.array([i-lam*j for i,j in zip(d,w)]))

        #shift and normalize
        p_list = [(p_list[i]-min(p_list[i]))/(max(p_list[i])-min(p_list[i])) for i in range(len(p_list))]

        #shit s.t. average at 0
        p_list = [-sum(p_list[i])/len(p_list[i])+p_list[i] for i in range(len(p_list))]
        
        chi_list = [statistics.variance(p_list[i]) for i in range(len(p_list))]

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
    def graph_gen(self):
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
        max_w = max(reduced_w)+1

        while len(V_prime) != 0:
            i_min = np.argmin(reduced_w)
            I.append(i_min)
            self._remove(V_prime, i_min)
            reduced_w[i_min] = max_w
            for n in list(nx.all_neighbors(graph, i_min)):
                self._remove(V_prime, n)
                reduced_w[n] = max_w

        w_tot = 0
        for i in I:
            w_tot += w[i]

        return I, w_tot
    
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
        dev2 = -np.diff(dev1)/np.diff(lam_list[:-1])
        m_index = np.array(argrelextrema(dev2, np.greater))[0]/100

        return dev1, dev2, m_index


