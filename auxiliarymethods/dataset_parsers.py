import graph_tool as gt
import numpy as np
import os.path as path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from datetime import datetime
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import TomekLinks


def read_txt(ds_name):
    pre = ""

    with open("datasets/" + pre + ds_name + "/" + ds_name + "_graph_indicator.txt", "r") as f:
        graph_indicator = [int(i) - 1 for i in list(f)]
    f.closed

    # Nodes
    num_graphs = max(graph_indicator)
    node_indices = []
    offset = []
    c = 0

    for i in range(num_graphs + 1):
        offset.append(c)
        c_i = graph_indicator.count(i)
        node_indices.append((c, c + c_i - 1))
        c += c_i

    graph_db = []
    vertex_list = []
    for i in node_indices:
        g = gt.Graph(directed=False)
        vertex_list_g = []
        for _ in range(i[1] - i[0] + 1):
            vertex_list_g.append(g.add_vertex())

        graph_db.append(g)
        vertex_list.append(vertex_list_g)

    # Edges
    with open("datasets/" + pre + ds_name + "/" + ds_name + "_A.txt", "r") as f:
        edges = [i.split(',') for i in list(f)]
    f.closed

    edges = [(int(e[0].strip()) - 1, int(e[1].strip()) - 1) for e in edges]

    edge_indicator = []
    edge_list = []
    for e in edges:
        g_id = graph_indicator[e[0]]
        edge_indicator.append(g_id)
        g = graph_db[g_id]
        off = offset[g_id]

        # Avoid multigraph
        if not g.edge(e[0] - off, e[1] - off):
            edge_list.append(g.add_edge(e[0] - off, e[1] - off))

    # Node labels
    if path.exists("datasets/" + pre + ds_name + "/" + ds_name + "_node_labels.txt"):
        with open("datasets/" + pre + ds_name + "/" + ds_name + "_node_labels.txt", "r") as f:
            node_labels = [int(i) for i in list(f)]
        f.closed

        i = 0
        for g in graph_db:
            g.vp.nl = g.new_vertex_property("int")
            for v in g.vertices():
                g.vp.nl[v] = node_labels[i]
                i += 1


    # Node Attributes
    if path.exists("datasets/" + pre + ds_name + "/" + ds_name + "_node_attributes.txt"):
        with open("datasets/" + pre + ds_name + "/" + ds_name + "_node_attributes.txt", "r") as f:
            node_attributes = [map(float, i.split(',')) for i in list(f)]
        f.closed

        i = 0
        for g in graph_db:
            g.vp.na = g.new_vertex_property("vector<float>")
            for v in g.vertices():
                g.vp.na[v] = node_attributes[i]
                i += 1


    # Edge Labels
    if path.exists("datasets/" + ds_name + "/" + ds_name + "_edge_labels.txt"):
        with open("datasets/" + ds_name + "/" + ds_name + "_edge_labels.txt", "r") as f:
            edge_labels = [int(i) for i in list(f)]
        f.closed

        l_el = []
        for i in range(num_graphs + 1):
            g = graph_db[graph_indicator[i]]
            l_el.append(g.new_edge_property("int"))

        for i, l in enumerate(edge_labels):
            g_id = edge_indicator[i]
            g = graph_db[g_id]

            l_el[g_id][edge_list[i]] = l
            g.ep.el = l_el[g_id]

    # Edge Attributes
    if path.exists("datasets/" + ds_name + "/" + ds_name + "_edge_attributes.txt"):
        with open("datasets/" + ds_name + "/" + ds_name + "_edge_attributes.txt", "r") as f:
            edge_attributes = [map(float, i.split(',')) for i in list(f)]
        f.closed

        l_ea = []
        for i in range(num_graphs + 1):
            g = graph_db[graph_indicator[i]]
            l_ea.append(g.new_edge_property("vector<float>"))

        for i, l in enumerate(edge_attributes):
            g_id = edge_indicator[i]
            g = graph_db[g_id]

            l_ea[g_id][edge_list[i]] = l
            g.ep.ea = l_ea[g_id]

    # Classes
    with open("datasets/" + pre + ds_name + "/" + ds_name + "_graph_labels.txt", "r") as f:
        classes = [int(i) for i in list(f)]
    f.closed

    positive_graphs = []
    positive_labels = []
    negative_graphs = []
    negative_labels = []
    
    for index, graph in enumerate(graph_db):
        if classes[index] != 0:
            positive_graphs.append(graph)
            positive_labels.append(classes[index])
    
    for index, graph in enumerate(graph_db):
        if len(negative_graphs) > (len(positive_graphs) * 1.2):
            break
        if classes[index] == 0:
            negative_graphs.append(graph)
            negative_labels.append(classes[index])
        
    print("Negative graphs size: " + str(len(negative_graphs)) + ", positive graphs size: " + str(len(positive_graphs)))
    return negative_graphs + positive_graphs, negative_labels + positive_labels


def write_lib_svm(gram_matrix, classes, name, exec_time):
    with open(name, "w") as f:
        k = 1
        for c, row in zip(classes, gram_matrix):
            s = ""
            s = str(c) + " " + "0:" + str(k) + " "
            for i, r in enumerate(row):
                s += str(i + 1) + ":" + str(r) + " "
            s += "\n"
            f.write(s)
            k += 1
    f.closed
    
    
    # cc = ClusterCentroids(sampling_strategy='majority', n_jobs=-1, random_state=42)
    # X_cc, y_cc = cc.fit_sample(gram_matrix, classes)


    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        gram_matrix, classes, train_size=0.8, test_size=0.2, random_state=42)

    test_zeros = y_test.count(0)
    test_ones = test_zeros * 0.15
    
    while y_test.count(1) > test_ones:
        for index, label in y_test:
            if label == 1:
                graph = X_test.pop(index)
                insert_label = y_test.pop(index)
                X_train.append(graph)
                y_train.append(insert_label)
                break
    
    print("X_train: " + str(len(X_train)))
    print("y_train: " + str(len(y_train)))
    print("X_test: " + str(len(X_test)))
    print("y_test: " + str(len(y_test)))
   
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 50]},
                       {'kernel':['poly'], 'degree':[2,3]}]

    clf = GridSearchCV(SVC(class_weight='balanced'), tuned_parameters, cv=10, scoring="f1_weighted", n_jobs=-1)
    clf.fit(X_train, y_train)
    report_str="Detailed classification report:\n\n"
    report_str += ("The model is trained on the full development set.\n")
    report_str += ("Total execution time: " + str(exec_time) + ".\n")
    report_str += ("The scores are computed on the full evaluation set.\n\n")
    y_true, y_pred = y_test, clf.predict(X_test)
    report_str += "\n" + (classification_report(y_true, y_pred))
    print(report_str)
    with open(name.replace('_gram_matrix', '') + "_result.txt", "w") as text_file:
        text_file.write(report_str)
