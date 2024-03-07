import os
from typing import TypeVar, Optional, Callable
from enum import Enum
import bisect
from os import get_terminal_size

T = TypeVar("T")


class VisitStatus(Enum):
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
        self.__status: VisitStatus = VisitStatus.NO_VISITED
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
    def status(self) -> VisitStatus:
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

    def print_graph(self):
        for node in sorted(list(self.__vertex), key=lambda n: n.value):
            nb = [*map(lambda n: f"\033[0;32m{n[0]}\033[0;37m: \033[0;36m{n[1]}\033[0;37m", node.neighbors.items())]
            print(f"{node.value} -> {{{', '.join(nb)}}}")

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
            node.status = VisitStatus.VISITED
            result.append(node)

            for n in [n for n in node.neighbors if n is not n.status == VisitStatus.NO_VISITED]:
                queue.append(n)
                path.append([*node_path, n])

            if stop_condition and stop_condition(node.value):
                result = node_path
                break

        self.__reset_vertex_status()

        return result

    def __reset_vertex_status(self) -> None:
        """
        Reset all the nodes to No Visited
        """
        for vertex in self.__vertex:
            vertex.status = VisitStatus.NO_VISITED

    @classmethod
    def __dfs_visit(cls, vertex: GraphNode, visited_order: list[GraphNode],
                    target: Optional[Callable[[T], bool]] = None) -> bool:
        """Method to visit a node and its neighbors if some of those were not visited"""
        vertex.status = VisitStatus.VISITED
        cls.__global_time += 1
        vertex.distance = cls.__global_time
        visited_order.append(vertex)

        if target and target(vertex.value):
            return True

        for v in vertex.neighbors:
            if v.status == VisitStatus.NO_VISITED:
                v.parent = vertex
                if cls.__dfs_visit(v, visited_order, target):
                    cls.__global_time += 1
                    v.final_distance = cls.__global_time
                    return True

        vertex.status = VisitStatus.FINISHED
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
            vertex.status = VisitStatus.NO_VISITED
            vertex.parent = None

        self.__global_time = 0

        # Start from the root node
        if self.__dfs_visit(self.__root, visit_order, target):
            self.__reset_vertex_status()
            return visit_order

        for vertex in self.__vertex:
            if vertex.status == VisitStatus.NO_VISITED:
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

            node.status = VisitStatus.VISITED

            for n in [n for n in node.neighbors if n.status == VisitStatus.NO_VISITED]:
                result = self.dfs_limited(n, target, depth - 1)
                if result:
                    result.insert(0, node)
                    self.__reset_vertex_status()
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
                    self.__reset_vertex_status()
                    return result
            depth += 1

    def uniform_cost_search(self, root: GraphNode, target: Callable[[T], bool]):
        """
        This algorithm uses a priority list and a explored set to determine which nodes to expand, on the priority list
        it always will have the minor cost path at the first element, due to the use of bisect.insort to place the
        minor cost path at the start of the list.
        :param root: Start Node (or root node by default).
        :param target: Function with the condition to mark a node as the target.
        :return: Minor cost path.
        """
        if not root:
            root = self.root

        frontier: list[tuple[list[GraphNode], int]] = [([root, ], 0)]
        explored: set[GraphNode] = set()

        while True:
            if not frontier:
                return None

            path, cost = frontier.pop(0)
            node = path[-1]

            explored.add(node)

            if target(node.value):
                return path

            for n in node.neighbors:
                if n not in explored:
                    new_cost = cost + node.neighbors[n]
                    new_path = [*path, n]
                    bisect.insort(frontier, (new_path, new_cost), key=lambda t: t[1])

    @classmethod
    def get_node_list_distance(cls, node_list: list[GraphNode]) -> int:
        """
        Calculates the distance traveled of a node path.
        :param node_list: List representing the node path.
        :return: Total cost traveled.
        """
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


def print_results(path: list[GraphNode]) -> None:
    print(f"Camino: {path}")
    path_distance = Graph.get_node_list_distance(path)
    print(f"Distancia: {path_distance}")


def get_path(nodo_inicio: GraphNode, nodo_fin: GraphNode, num_ruta: int = 1) -> None:
    inicio = nodo_inicio.value
    objetivo = nodo_fin.value
    print(f"[ Ruta {num_ruta} ]".center(term_width, 'o'))
    print(f"> Nodo de inicio: {inicio}")
    print(f"> Nodo objetivo: {objetivo}")

    print("[ Breadth First Search ]".center(term_width, '='))
    result_path = edc.breadth_first_search(lambda n: n == objetivo)
    print_results(result_path)

    print("[ Depth First Search ]".center(term_width, '='))
    result_path = edc.depth_first_search(lambda n: n == objetivo)
    print_results(result_path)

    print("[ Depth First Search Limitado ]".center(term_width, '='))
    result_path = edc.dfs_limited(edc.root, lambda n: n == objetivo, 6)
    print_results(result_path)

    print("[ Depth First Search Iterativo ]".center(term_width, '='))
    result_path = edc.dfs_iterative(edc.root, lambda n: n == objetivo, 6)
    print_results(result_path)

    print("[ Búsqueda de Costo Uniforme ]".center(term_width, '='))
    result_path = edc.uniform_cost_search(edc.root, lambda n: n == objetivo)
    print_results(result_path)


# Aplicar DFS y BFS para encontrar la ruta a seguir desde la entrada hacia los escenarios
# “Kinetic Field” y “circuit grounds”.
if __name__ == '__main__':
    try:
        term_width = os.get_terminal_size().columns
    except OSError:
        term_width = 80

    print("".center(term_width, '-'))
    print(f"|{'Facultad de Ingeniería - UNAM'.center(term_width - 2, ' ')}|")
    print(f"|{'Inteligencia Artificial - Semestre 2024-2'.center(term_width - 2, ' ')}|")
    print("".center(term_width, '-'))

    A = GraphNode("Entrada")
    B = GraphNode("Recarga Info")
    C = GraphNode("XX Stage")
    D = GraphNode("Servicios")
    E = GraphNode("Circuit Grounds")
    F = GraphNode("Pixel Forest")
    G = GraphNode("Forest Jungle")
    H = GraphNode("Cantina del centro")
    I = GraphNode("Bebidas")
    J = GraphNode("Kinetic")
    K = GraphNode("Surtidora Sur")
    edc = Graph(A, {A, B, C, D, E, F, G, H, I, J, K})

    edc.add_edge(A, B, 5)
    edc.add_edge(A, C, 8)
    edc.add_edge(B, C, 8)
    edc.add_edge(C, D, 2)
    edc.add_edge(D, B, 3)
    edc.add_edge(D, E, 4)
    edc.add_edge(E, I, 7)
    edc.add_edge(E, H, 11)
    edc.add_edge(F, E, 10)
    edc.add_edge(F, I, 6)
    edc.add_edge(F, G, 5)
    edc.add_edge(F, C, 7)
    edc.add_edge(G, I, 6)
    edc.add_edge(G, K, 5)
    edc.add_edge(K, J, 9)
    edc.add_edge(H, I, 6)
    edc.add_edge(H, J, 8)
    edc.add_edge(I, J, 15)

    print("[ Prueba de algoritmos búsqueda en grafos ]".center(term_width, '='))

    print("Grafo: ")
    edc.print_graph()

    get_path(A, J, 1)
    get_path(A, E, 2)
