import heapq
import os
from enum import Enum
from typing import TypeVar, Optional, Callable

# noinspection DuplicatedCode
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
        self.__g = -1
        self.__h = -1
        self.__f = -1

    @property
    def parent(self):
        return self.__parent

    @parent.setter
    def parent(self, value):
        self.__parent = value

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
    def status(self) -> VisitStatus:
        return self.__status

    @status.setter
    def status(self, value: int) -> None:
        self.__status = value

    @property
    def g(self):
        return self.__g

    @g.setter
    def g(self, value):
        self.__g = value

    @property
    def h(self):
        return self.__h

    @h.setter
    def h(self, value):
        self.__h = value

    @property
    def f(self):
        return self.__f

    @f.setter
    def f(self, value):
        self.__f = value

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

    def print_graph(self) -> None:
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

    @classmethod
    def calc_heuristic(cls, start: GraphNode, target: GraphNode):
        return start.neighbors.get(target, 0)

    def a_star(self, start: GraphNode, target: GraphNode) -> Optional[list[GraphNode]]:
        close_nodes = []
        open_nodes: set[GraphNode] = {start}

        start.g = 0
        start.h = self.calc_heuristic(start, target)
        start.f = start.g + start.h

        while open_nodes:
            current = min(open_nodes, key=lambda n: n.f)
            if current == target:
                path = []
                while current:
                    path.append(current)
                    current = current.parent
                return path[::-1]

            open_nodes.remove(current)
            close_nodes.append(current)

            for vecino in current.neighbors:
                if vecino in close_nodes:
                    continue

                new_cost = current.g + current.neighbors[vecino]  # costo actual del camino hasta el vecino
                if vecino not in open_nodes or new_cost < vecino.g:
                    vecino.g = new_cost
                    vecino.h = self.calc_heuristic(vecino, target)
                    vecino.f = vecino.g + vecino.h
                    vecino.parent = current
                    if vecino not in open_nodes:
                        open_nodes.add(vecino)

        return None


def test():
    sn = GraphNode("S")
    an = GraphNode("A")
    bn = GraphNode("B")
    cn = GraphNode("C")
    dn = GraphNode("D")
    en = GraphNode("E")
    fn = GraphNode("F")
    gn = GraphNode("G")
    inn = GraphNode("I")
    xn = GraphNode("X")
    graph = Graph(sn, {sn, an, bn, cn, dn, en, fn, gn, inn, xn})

    graph.add_edge(sn, an, 5)
    graph.add_edge(sn, bn, 9)
    graph.add_edge(sn, cn, 6)
    graph.add_edge(sn, dn, 6)
    graph.add_edge(an, bn, 3)
    graph.add_edge(bn, cn, 1)
    graph.add_edge(cn, dn, 2)
    graph.add_edge(an, gn, 9)
    graph.add_edge(bn, xn, 7)
    graph.add_edge(cn, inn, 5)
    graph.add_edge(cn, fn, 7)
    graph.add_edge(fn, inn, 5)
    graph.add_edge(inn, xn, 3)
    graph.add_edge(dn, en, 2)
    graph.add_edge(en, xn, 3)

    graph.print_graph()

    path = graph.a_star(sn, xn)
    print(path)

    exit(0)


if __name__ == '__main__':
    test()
    # noinspection DuplicatedCode
    try:
        term_width = os.get_terminal_size().columns
    except OSError:
        term_width = 80

    print("".center(term_width, '-'))
    print(f"|{'Facultad de Ingeniería - UNAM'.center(term_width - 2, ' ')}|")
    print(f"|{'Inteligencia Artificial - Semestre 2024-2'.center(term_width - 2, ' ')}|")
    print(f"|{'Práctica 3 - Búsqueda informada y búsqueda local'.center(term_width - 2, ' ')}|")
    print("".center(term_width, '-'))
