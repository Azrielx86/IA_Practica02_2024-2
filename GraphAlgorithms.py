import math
import time
from typing import TypeVar, Optional, Callable
from enum import Enum
from random import randint

T = TypeVar("T")


class GraphVisitStatus(Enum):
    """Used to declare the visited status of a node."""
    VISITED = 0
    NO_VISITED = 1
    FINISHED = 2


class GraphNode:
    """Node class for a graph"""

    def __init__(self, value: T) -> None:
        """
        :param value: Data to store in the node
        """
        self.__value: T = value
        self.__neighbours: dict[GraphNode, int] = {}
        self.__status: GraphVisitStatus = GraphVisitStatus.NO_VISITED
        self.__parent: Optional[GraphNode] = None
        self.__distance: int = -1
        self.__final_distance: int = -1

    @property
    def parent(self):
        return self.__parent

    @parent.setter
    def parent(self, value):
        self.__parent = value

    @property
    def final_distance(self) -> int:
        return self.__final_distance

    @final_distance.setter
    def final_distance(self, value: int) -> None:
        self.__final_distance = value

    @property
    def neighbors(self) -> dict["GraphNode", int]:
        return self.__neighbours

    @neighbors.setter
    def neighbors(self, value):
        self.__neighbours = value

    @property
    def value(self) -> T:
        return self.__value

    @value.setter
    def value(self, value: T) -> None:
        self.__value = value

    @property
    def distance(self) -> int:
        return self.__distance

    @distance.setter
    def distance(self, value: int) -> None:
        self.__distance = value

    @property
    def status(self) -> GraphVisitStatus:
        return self.__status

    @status.setter
    def status(self, value: int) -> None:
        self.__status = value

    def add_neighbour(self, node: "GraphNode", dist: int):
        """Adds a node to the neighbors list"""
        if node not in self.__neighbours.keys():
            self.__neighbours[node] = dist
        else:
            print(f"Node {node} already in {self} neighbors list")

    def __str__(self) -> str:
        return str(self.__value)

    def __repr__(self) -> str:
        return self.__str__()


class Graph:
    __global_time: int = -1

    def __init__(self, root: Optional[GraphNode] = None, vertex: Optional[set[GraphNode]] = None) -> None:
        self.__root: GraphNode | None = root
        self.__vertex: set[GraphNode] = vertex if vertex else set()

    @property
    def vertex(self) -> set[GraphNode]:
        return self.__vertex

    @property
    def root(self) -> GraphNode:
        return self.__root

    @root.setter
    def root(self, value: GraphNode) -> None:
        self.__root = value

    def add_edge(self, node1: GraphNode, node2: GraphNode, weight: int) -> None:
        """Adds an edge to the graph, and adds both nodes to their respective neighbors list."""
        node1.add_neighbour(node2, weight)
        node2.add_neighbour(node1, weight)

        if node1 not in self.__vertex:
            self.__vertex.add(node1)

        if node2 not in self.__vertex:
            self.__vertex.add(node2)

    def breadth_first_search(self, stop_condition: Optional[Callable[[T], bool]] = None) -> list[GraphNode]:
        """
        How the algorithm works:
            1. Start at the root node and add it to a queue.
            2. While the queue is not empty, dequeue a node and visit it.
            3. Enqueue all of its children (if any) into the queue.
            4. Repeat steps 2 and 3 until the queue is empty.
        :param stop_condition: Condition to search a node and stop the algorithm
        :return: A list with the traveled nodes
        """
        queue = [self.__root, ]
        path = [[self.__root, ], ]
        result = []

        while len(queue) > 0:
            node = queue.pop(0)
            node_path = path.pop(0)
            node.status = GraphVisitStatus.VISITED
            result.append(node)

            for n in [n for n in node.neighbors if n is not n.status == GraphVisitStatus.NO_VISITED]:
                queue.append(n)
                path.append([*node_path, n])

            if stop_condition and stop_condition(node.value):
                result = node_path
                break

        self.__reset_vertex_status()

        return result

    def __reset_vertex_status(self):
        for vertex in self.__vertex:
            vertex.status = GraphVisitStatus.NO_VISITED

    @classmethod
    def __dfs_visit(cls, vertex: GraphNode, visited_order: list[GraphNode],
                    target: Optional[Callable[[T], bool]] = None) -> bool:
        """Method to visit a node and its neighbors if some of those were not visited"""
        vertex.status = GraphVisitStatus.VISITED
        cls.__global_time += 1
        vertex.distance = cls.__global_time
        visited_order.append(vertex)

        if target and target(vertex.value):
            return True

        for v in vertex.neighbors:
            if v.status == GraphVisitStatus.NO_VISITED:
                v.parent = vertex
                if cls.__dfs_visit(v, visited_order, target):
                    cls.__global_time += 1
                    v.final_distance = cls.__global_time
                    return True

        vertex.status = GraphVisitStatus.FINISHED
        cls.__global_time += 1
        vertex.final_distance = cls.__global_time
        return False

    def depth_first_search(self, target: Optional[Callable[[T], bool]] = None) -> list[GraphNode]:
        """
        Depth First Search: Recursive approach.
        :param target: Function to found a node and stop the algorithm.
        :return: Path to the target node from the root node.
        """
        visit_order = []
        for vertex in self.__vertex:
            vertex.status = GraphVisitStatus.NO_VISITED
            vertex.parent = None

        self.__global_time = 0

        # Start from the root node
        if self.__dfs_visit(self.__root, visit_order, target):
            self.__reset_vertex_status()
            return visit_order

        for vertex in self.__vertex:
            if vertex.status == GraphVisitStatus.NO_VISITED:
                if self.__dfs_visit(vertex, visit_order, target):
                    self.__reset_vertex_status()
                    return visit_order

        self.__reset_vertex_status()
        return visit_order

    def dfs_limited(self, node: GraphNode, target: Callable[[T], bool], depth: int) -> Optional[list[GraphNode]]:
        """
        Limited DFS Implementation
        :param node: Start node of the algorithm
        :param target: Function with the condition to found the target and stop the algorithm
        :param depth: Max depth to search the node.
        :return: Path from the start node to the target node
        """
        if depth >= 0:
            if target(node.value):
                return [node, ]

            node.status = GraphVisitStatus.VISITED

            for n in [n for n in node.neighbors if n.status == GraphVisitStatus.NO_VISITED]:
                result = self.dfs_limited(n, target, depth - 1)
                if result:
                    result.insert(0, node)
                    return result
        return None

    def __dfs_lim_iter(self, node: GraphNode, target: Callable[[T], bool], depth: int) -> Optional[list[GraphNode]]:
        """Helper function to the DFS iterative implementation"""
        if depth >= 0:
            if target(node.value):
                return [node, ]

            for n in node.neighbors:
                result = self.__dfs_lim_iter(n, target, depth - 1)
                if result:
                    result.insert(0, node)
                    return result
        return None

    def dfs_iterative(self, root: GraphNode, target: Callable, max_depth: int = 0) -> Optional[list[GraphNode]]:
        """
        Iterative Depth First Search Implementation
        :param root: Root node of the graph or start point to search
        :param target: Function with the condition to stop the algorithm
        :param max_depth: Max depth to search and stop the algorithm
        :return:
        """
        depth = 0
        while depth <= max_depth:
            result = self.__dfs_lim_iter(root, target, depth)
            if result:
                if target and target(result[-1].value):
                    return result
            depth += 1

    # def uniform_cost_search(self, root: GraphNode, target: Callable[[T], bool]):
    #     frontier: list[tuple[GraphNode, int]] = [(root, 0)]
    #     explored: set[GraphNode] = set()
    #
    #     while True:
    #         if not frontier:
    #             return None
    #
    #         node, cost = frontier.pop(0)
    #
    #         if target(node.value):
    #             return node
    #
    #         explored.add(node)
    #
    #         for n in node.neighbors:
    #             if n not in explored:

    @classmethod
    def get_node_list_distance(cls, node_list: list[GraphNode]):
        if len(node_list) < 1:
            return -1
        total_cost = 0
        node, next_node = node_list.pop(0), node_list.pop(0)
        while node_list:
            total_cost += node.neighbors[next_node]

            node = next_node
            next_node = node_list.pop(0)
        total_cost += node.neighbors[next_node]

        return total_cost


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

    graph.add_edge(node, left_child, randint(0, 50))
    graph.add_edge(node, right_child, randint(0, 50))

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

    # tree.uniform_cost_search(tree.root, lambda x: x == 30)


    # Del proyecto
    A = GraphNode("A(Entrada docker)")
    B = GraphNode("B(Recarga info)")
    C = GraphNode("C(Dos X Stage)")
    D = GraphNode("D(Servicios)")
    E = GraphNode("E(Circuit G)")
    F = GraphNode("F(Pixel Forest)")
    G = GraphNode("G(Forest Jungle)")
    H = GraphNode("H(Contra???)")
    I = GraphNode("I(Bebidas)")
    J = GraphNode("J(Kinetic)")
    K = GraphNode("K(Surtidora?)")

    A.add_neighbour(B, 5)
    A.add_neighbour(C, 8)
    B.add_neighbour(C, 8)
    C.add_neighbour(B, 3)
    C.add_neighbour(D, 2)
    D.add_neighbour(B, 3)
    D.add_neighbour(E, 4)
    E.add_neighbour(I, 7)
    E.add_neighbour(H, 11)
    F.add_neighbour(E, 10)
    F.add_neighbour(I, 6)
    F.add_neighbour(G, 5)
    G.add_neighbour(I, 6)
    G.add_neighbour(K, 5)
    K.add_neighbour(J, 9)
    H.add_neighbour(I, 6)
    H.add_neighbour(J, 8)
    H.add_neighbour(E, 11)

    EDC = Graph(A, {A, B, C, D, E, F, G, H, I, J})

    # Ejemplo para obtener la distancia de un recorrido :P
    a_to_j = EDC.breadth_first_search(lambda n: n[0] == 'J')
    dist = Graph.get_node_list_distance(a_to_j.copy() if a_to_j else [])
    print(f"{a_to_j} | Distance = {dist}")
    a_to_j = EDC.dfs_limited(EDC.root, lambda n: n[0] == 'J', 15) # Sospecho que este hay que corregirlo
    dist = Graph.get_node_list_distance(a_to_j.copy() if a_to_j else [])
    print(f"{a_to_j} | Distance = {dist}")
