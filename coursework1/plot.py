
import pydot

# def draw(tree_graph,parent_name, child_name):
#     edge = pydot.Edge(parent_name, child_name)
#     tree_graph.add_edge(edge)
def visit(tree_graph,node, parent=None):
    global i
    attribute= list(node.keys())[0]
    if parent is None:
        from_name = str(i)
        i += 1
        from_label = "Root"
        node_from = pydot.Node(from_name, label=from_label)
        tree_graph.add_node(node_from)
        visit(tree_graph, node, node_from)

    for split_value, split_reslut in node[attribute].items():
        if parent:
            if isinstance(split_reslut, dict):
                # We start with the root node whose parent is None
                # we don't want to graph the None node
                from_name = str(i)
                from_label = "No. attribute: " + str(attribute) + " Conditions: " + str(split_value)
                i += 1
                node_from = pydot.Node(from_name, label=from_label)
                tree_graph.add_node(node_from)
                tree_graph.add_edge(pydot.Edge(parent, node_from))
                visit(tree_graph, split_reslut, node_from)
            ## if we encounter the leaves
            else:
                from_name = str(i)
                from_label ="No. attribute: " + str(attribute) + " Conditions: " + str(split_value)
                i += 1
                node_from = pydot.Node(from_name, label=from_label)
                tree_graph.add_node(node_from)
                tree_graph.add_edge(pydot.Edge(parent, node_from))
                to_name=str(i)# unique name
                i+=1
                to_label = split_reslut
                node_to = pydot.Node(to_name, label=to_label, shape='box')
                tree_graph.add_node(node_to)
                tree_graph.add_edge(pydot.Edge(node_from, node_to))




def plot_tree(node):
    tree_graph = pydot.Dot(graph_type='graph')
    visit(tree_graph,node)
    tree_graph.write_png('example1_graph.png')
i=0
