import heapq
import os
from enum import Enum
from typing import TypeVar, Optional, Callable
from math import inf

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
        # for node in sorted(list(self.__vertex), key=lambda n: n.value):
        for node in list(self.__vertex):
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
    def a_star(cls, start: GraphNode, target: GraphNode, calc_heuristic: Callable[[GraphNode, GraphNode], int]) -> \
            Optional[list[GraphNode]]:
        close_nodes = []
        open_nodes: set[GraphNode] = {start}

        start.g = 0
        start.h = calc_heuristic(start, target)
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
                    vecino.h = calc_heuristic(vecino, target)
                    vecino.f = vecino.g + vecino.h
                    vecino.parent = current
                    if vecino not in open_nodes:
                        open_nodes.add(vecino)

        return None

    def ascenso_colina(self, inicio: GraphNode, get_value: Callable[[GraphNode], int]):
        actual = inicio
        mejor_vecino = GraphNode(None)
        while mejor_vecino:
            mejor_vecino = None
            mejor_valor = get_value(actual)
            for v in actual.neighbors:
                if get_value(v) < mejor_valor:
                    mejor_vecino = v
                    mejor_valor = get_value(v)
            if mejor_vecino is None:
                return actual
            actual = mejor_vecino
        return actual


class Sitio:
    def __init__(self, nombre: str, interes: int) -> None:
        self.__nombre: str = nombre
        self.__interes: int = interes

    @property
    def nombre(self) -> str:
        return self.__nombre

    @nombre.setter
    def nombre(self, value):
        self.__nombre = value

    @property
    def interes(self) -> int:
        return self.__interes

    @interes.setter
    def interes(self, value):
        self.__interes = value

    def __str__(self) -> str:
        return f"{self.__nombre} | Interés {self.__interes}"

    def __repr__(self) -> str:
        return self.__str__()

# def heuristica_pc(start: GraphNode, target: GraphNode) -> int:
#     while node := target.parent:



def test_ac():
    an = GraphNode(Sitio("A", 10))
    bn = GraphNode(Sitio("B", 10))
    cn = GraphNode(Sitio("C", 2))
    dn = GraphNode(Sitio("D", 4))
    en = GraphNode(Sitio("E", 5))
    fn = GraphNode(Sitio("F", 7))
    gn = GraphNode(Sitio("G", 3))
    inn = GraphNode(Sitio("I", 6))
    kn = GraphNode(Sitio("K", 0))
    graph = Graph(an, {an, bn, cn, dn, en, fn, gn, inn, kn})

    graph.add_edge(an, bn, 0)
    graph.add_edge(bn, dn, 0)
    graph.add_edge(bn, cn, 0)
    graph.add_edge(cn, kn, 0)
    graph.add_edge(an, fn, 0)
    graph.add_edge(fn, en, 0)
    graph.add_edge(fn, gn, 0)
    graph.add_edge(en, inn, 0)
    graph.add_edge(inn, kn, 0)

    graph.print_graph()

    res = graph.ascenso_colina(an, lambda n: n.value.interes)
    print(res)

    exit(0)


def test_a_star():
    sn = GraphNode(Sitio("S", 0))
    an = GraphNode(Sitio("A", 0))
    bn = GraphNode(Sitio("B", 0))
    cn = GraphNode(Sitio("C", 0))
    dn = GraphNode(Sitio("D", 0))
    en = GraphNode(Sitio("E", 0))
    fn = GraphNode(Sitio("F", 0))
    gn = GraphNode(Sitio("G", 0))
    inn = GraphNode(Sitio("I", 0))
    xn = GraphNode(Sitio("X", 0))
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

    path = graph.a_star(sn, xn, lambda s, t: 0)
    print(path)

    exit(0)


if __name__ == '__main__':
    test_ac()
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
