import networkx as nx

def build_character_graph(doc):

    G = nx.Graph()

    for sent in doc.sents:

        names = [
            ent.text
            for ent in sent.ents
            if ent.label_ == "PERSON"
        ]

        for i in range(len(names)):
            for j in range(i+1,len(names)):

                if G.has_edge(names[i],names[j]):
                    G[names[i]][names[j]]["weight"] += 1
                else:
                    G.add_edge(names[i],names[j],weight=1)

    return G