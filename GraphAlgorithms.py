import math
from typing import TypeVar, Optional, Callable
from enum import Enum

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

    def __init__(self) -> None:
        self.__root: GraphNode | None = None
        self.__vertex: set[GraphNode] = set()

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

    def breadth_first_search(self, stop_condition: Optional[Callable[[T], bool]]) -> list[GraphNode]:
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
        result = []

        while len(queue) > 0:
            node = queue.pop(0)
            node.status = GraphVisitStatus.VISITED
            result.append(node)

            for n in [n for n in node.neighbors if n is not n.status == GraphVisitStatus.NO_VISITED]:
                queue.append(n)

            if stop_condition and stop_condition(node.value):
                break

        for n in self.__vertex:
            n.status = GraphVisitStatus.NO_VISITED

        return result

    @classmethod
    def __dfs_visit(cls, vertex: GraphNode, visited_order: list[GraphNode]) -> int:
        """Method to visit a node and its neighbors if some of those were not visited"""
        vertex.status = GraphVisitStatus.VISITED
        cls.__global_time = cls.__global_time + 1
        vertex.distance = cls.__global_time
        visited_order.append(vertex)

        for v in vertex.neighbors:
            if v.status == GraphVisitStatus.NO_VISITED:
                v.parent = vertex
                cls.__dfs_visit(v, visited_order)

        vertex.status = GraphVisitStatus.FINISHED
        cls.__global_time = cls.__global_time + 1
        vertex.final_distance = cls.__global_time
        return cls.__global_time

    def depth_first_search(self) -> list[GraphNode]:
        """Depth First Search: Recursive approach."""
        visit_order = []
        for vertex in self.__vertex:
            vertex.status = GraphVisitStatus.NO_VISITED
            vertex.parent = None

        self.__global_time = 0

        # Start from the root node
        self.__dfs_visit(self.__root, visit_order)

        for vertex in self.__vertex:
            if vertex.status == GraphVisitStatus.NO_VISITED:
                self.__dfs_visit(vertex, visit_order)
        return visit_order

    def dfs_limited(self, node: GraphNode, target: Callable[[T], bool], depth: int) -> Optional[GraphNode]:
        if depth >= 0:
            if target(node.value):
                return node

            for n in node.neighbors:
                self.dfs_limited(n, target, depth - 1)

    def depth_first_search_iterative(self, root: GraphNode, target: Callable) -> Optional[GraphNode]:
        depth = 0
        while True:
            result = self.dfs_limited(root, target, depth)
            if target(result.value):
                return result
            depth += 1


def generateTree(graph: Graph, nodeData: T, level: int, maxLevel: int, weight: int = 0) -> GraphNode:
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

    if level > maxLevel:
        return node

    left_child = generateTree(graph, left, level + 1, maxLevel)
    right_child = generateTree(graph, right, level + 1, maxLevel)

    graph.add_edge(node, left_child, weight)
    graph.add_edge(node, right_child, weight)

    return node


if __name__ == '__main__':
    tree = Graph()
    tree.root = generateTree(tree, 25, 0, 5)
    print(f"BFS travel: {tree.breadth_first_search(lambda x: x == 30)}")
    print(f"DFS: {tree.depth_first_search()}")
    r = tree.dfs_limited(tree.root, lambda n: n == 30, 15)
    print(r)
