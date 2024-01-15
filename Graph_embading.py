import random
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import warnings
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
warnings.filterwarnings('ignore')

class Graph_embading:
    def __init__(self,graph):
        self.G = graph
        
    def save_graph(self,file_name):
        #initialze Figure
        graph = self.G
        G = graph
        plt.figure(num=None, figsize=(20, 20))
        plt.axis('off')
        fig = plt.figure(1)
        pos = nx.spring_layout(graph)
        color_map = []
        for node in G:
            layer0 = "Layer_0"
            layer1 = "Layer_1"
            layer2 = "Layer_2"

            if layer0 in node:
                color_map.append('blue')
            elif layer1 in node: 
                color_map.append('green') 
            else:
                color_map.append('red') 
        nx.draw_networkx_nodes(graph,pos,node_color=color_map)
        nx.draw_networkx_edges(graph,pos)
        nx.draw_networkx_labels(graph,pos)

        cut = 0.05
        xmax = cut * max(xx for xx, yy in pos.values())
        ymax = cut * max(yy for xx, yy in pos.values())
        plt.xlim(0, xmax)
        plt.ylim(0, ymax)

        plt.savefig(file_name,bbox_inches="tight")
        plt.show()
        del fig

        #Assuming that the graph g has nodes and edges entered
        #save_graph(G,"my_graph.pdf")
        
    def drwa_graph(self):
        # use one of the edge properties to control line thickness
        edgewidth = [ d['weight'] for (u,v,d) in self.G.edges(data=True)]
        # layout
        pos = nx.spring_layout(self.G)
        color_map = []
        for node in self.G:
            layer0 = "layer0"
            layer1 = "layer1"
            layer2 = "layer2"

            if layer0 in node:
                color_map.append('blue')
            elif layer1 in node: 
                color_map.append('green') 
            elif layer2 in node:
                color_map.append('red') 
            else:
                color_map.append('black')
        nx.draw_networkx_nodes(self.G, pos,node_size=100,node_color=color_map)
        nx.draw_networkx_edges(self.G, pos,)
        #nx.draw_networkx_edges(self.G, pos, width=edgewidth,edge_color=edgewidth)
        
    def draw_layer_graph(self,graph):
        G = graph
        # use one of the edge properties to control line thickness
        edgewidth = [ d['weight'] for (u,v,d) in G.edges(data=True)]
        # layout
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos,node_size=100)
        nx.draw_networkx_edges(G, pos,)
        #nx.draw_networkx_edges(G, pos, width=edgewidth,edge_color=edgewidth)
        
    def get_node_list(self):
        nodes_list = np.array(list(self.G.nodes()))
        node_name = nodes_list[:]
        return node_name
        
    def get_rando_mwalk_node(self,node, path_length): 
        random_walk = [node]

        for i in range(path_length-1):
            temp = list(self.G.neighbors(node))
            temp = list(set(temp) - set(random_walk))    
            if len(temp) == 0:
                break

            random_node = random.choice(temp)
            random_walk.append(random_node)
            node = random_node

        return random_walk
    
    def randome_walk_nodes(self,node_name):
        # get list of all nodes from the graph
        all_nodes = list(self.G.nodes())

        random_walks = []
        for n in (node_name):
            for i in range(5):
                random_walks.append(self.get_rando_mwalk_node(n,15))
        # count of sequences
        return random_walks
    
    def embed_graph(self,random_walks,embedding_size):
        # train skip-gram (word2vec) model
        model_w2v = Word2Vec(window = 4, sg = 1, hs = 0,
                         negative = 10, # for negative sampling
                         alpha=0.03, min_alpha=0.0007,
                         seed = 14)

        model_w2v.build_vocab(random_walks, progress_per=2)

        model_w2v.train(random_walks, total_examples = model_w2v.corpus_count, epochs=20, report_delay=1)
        return model_w2v
    
    def plot_embaded_graph(self,model,word_list):
        X = model.wv[word_list]

        # reduce dimensions to 2
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)


        plt.figure(figsize=(12,9))
        # create a scatter plot of the projection
        plt.scatter(result[:, 0], result[:, 1])
        for i, word in enumerate(word_list):
            plt.annotate(word, xy=(result[i, 0], result[i, 1]))

        plt.show()
