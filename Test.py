from GraphAlgorithms import *
import math
import os
import time


def generate_tree(graph: Graph, nodeData: T, level: int, maxLevel: int, weight: int = 0) -> GraphNode:
    """
    Generates a tree with the condition:
        - Left node: :math:`x-1`
        - Right node: :math:`\\sqrt{x}`
    :param graph: Main Graph to add the node
    :param weight: Weight of the node
    :param maxLevel: Max height of the tree
    :param nodeData: integer to the node
    :param level: Current level of the generated node in the tree
    :return: The last created node, if it is the first call to the function, then it will return the root node.
    """
    node = GraphNode(nodeData)
    left, right = nodeData + 1, int(math.sqrt(nodeData))

    if level >= maxLevel:
        return node

    left_child = generate_tree(graph, left, level + 1, maxLevel)
    right_child = generate_tree(graph, right, level + 1, maxLevel)

    graph.add_edge(node, left_child, left)
    graph.add_edge(node, right_child, right)

    return node


def test_algorithm(function: Callable, message: Optional[str] = None, *args, **kwargs) -> None:
    """
    Function to test the execution time of a function,
    :param function: Function to test
    :param message: An optional message that will appear at the start of the message result,
                    if not defined it will be the function name
    :param args: Optional arguments of the function
    :param kwargs: Optional keyword arguments of the function
    """
    start = time.perf_counter()
    ret = function(*args, **kwargs)
    end = time.perf_counter()
    print(f"{message if message is not None else function} [Exec Time = {end - start:0.5e}]: {ret}")


if __name__ == '__main__':
    tree = Graph()
    tree.root = generate_tree(tree, 25, 0, 5)
    test_algorithm(tree.breadth_first_search, "BFS", stop_condition=lambda x: x == 30)
    test_algorithm(tree.depth_first_search, "DFS", lambda x: x == 30)
    test_algorithm(tree.dfs_limited, "DFS Limited (depth = 5; target = 7) ", tree.root, lambda x: x == 30, 5)
    test_algorithm(tree.dfs_iterative, "DFS Iterative", tree.root, lambda x: x == 30, 5)
    print(tree.uniform_cost_search(tree.root, lambda x: x == 8))
    print(tree.uniform_cost_search(tree.root, lambda x: x == 30))
