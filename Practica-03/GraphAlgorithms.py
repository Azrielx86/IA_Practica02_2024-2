import os
from enum import Enum
from typing import TypeVar, Optional, Callable, Generic
from math import sqrt
from random import randint, sample

# noinspection DuplicatedCode
T = TypeVar("T")

interes = 100


class VisitStatus(Enum):
    """Used to declare the visited status of a node."""
    VISITED = 0
    NO_VISITED = 1
    FINISHED = 2


class GraphNode(Generic[T]):
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

    def add_edge(self, node1: GraphNode, node2: GraphNode, weight: int = 0) -> None:
        """Adds an edge to the graph, and adds both nodes to their respective neighbors list."""
        node1.add_neighbour(node2, weight)
        node2.add_neighbour(node1, weight)

        if node1 not in self.__vertex:
            self.__vertex.add(node1)

        if node2 not in self.__vertex:
            self.__vertex.add(node2)

    @classmethod
    def a_star(cls, start: GraphNode, target: GraphNode, calc_heuristic: Callable[[GraphNode, GraphNode], float]) -> \
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

    @classmethod
    def ascenso_colina(cls, inicio: GraphNode[T], get_value: Callable[[GraphNode[T]], int]) -> GraphNode[T]:
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
    def __init__(self, nombre: str, interes: int = 0, coordinates: tuple[float, float] = None) -> None:
        self.__nombre: str = nombre
        self.__interes: int = interes
        self.__coordinates: tuple[float, float] = coordinates or [0, 0]

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

    @property
    def coordinates(self) -> tuple[float, float]:
        return self.__coordinates

    @coordinates.setter
    def coordinates(self, value: tuple[float, float]) -> None:
        self.__coordinates = value

    def __str__(self) -> str:
        return f"{self.__nombre}"

    def __repr__(self) -> str:
        return self.__str__()


def heuristic_p03(start: GraphNode[Sitio], target: GraphNode[Sitio]) -> float:
    global interes
    pos1 = start.value.coordinates
    pos2 = target.value.coordinates
    interes_nodo = start.neighbors.get(target, 0)
    heuristica = sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2) + interes_nodo
    interes -= heuristica % 10
    return heuristica


if __name__ == '__main__':
    # noinspection DuplicatedCode
    try:
        term_width = os.get_terminal_size().columns
    except OSError:
        term_width = 120

    fi = GraphNode(Sitio("FI UNAM", 0, (0, 500)))
    n00 = GraphNode(Sitio("Auditorio Nacional", 40, (500, 700)))
    n01 = GraphNode(Sitio("Concierto Islas", 20, (450, 550)))
    n02 = GraphNode(Sitio("Palacio de los deportes", 60, (700, 450)))
    n03 = GraphNode(Sitio("Parque bicentenario", 50, (440, 460)))
    n04 = GraphNode(Sitio("Friki Plaza", 10, (520, 550)))
    n05 = GraphNode(Sitio("Cineteca Nacional", 70, (520, 20)))
    n06 = GraphNode(Sitio("Mercado Coyoacán", 10, (505, 20)))
    n07 = GraphNode(Sitio("Torre Latino", 80, (510, 450)))
    n08 = GraphNode(Sitio("Acuario Inbursa", 90, (440, 550)))
    n09 = GraphNode(Sitio("Chapultepec", 65, (490, 500)))
    n0a = GraphNode(Sitio("Soumaya", 45, (300, 550)))
    n0c = GraphNode(Sitio("Feria Aztlan", 48, (490, 550)))
    n0d = GraphNode(Sitio("Sushi Roll", 95, (500, 700)))
    n0e = GraphNode(Sitio("Burguer King", 15, (700, 600)))
    n0f = GraphNode(Sitio("KFC", 5, (700, 450)))
    n11 = GraphNode(Sitio("Tacos Champs", 0, (300, 505)))
    n12 = GraphNode(Sitio("Gorditas Mixcoac", 59, (490, 400)))
    n13 = GraphNode(Sitio("Domino's Pizza", 55, (500, 650)))
    n14 = GraphNode(Sitio("Liru sisa", 25, (495, 20)))
    n15 = GraphNode(Sitio("Pizza Perro Negro", 65, (700, 550)))
    n16 = GraphNode(Sitio("Fiesta Colonia Valle", 35, (505, 400)))
    n17 = GraphNode(Sitio("Casa Alemana", 100, (150, 550)))
    n18 = GraphNode(Sitio("Cata de bebidas en islas", 47, (300, 551)))
    n19 = GraphNode(Sitio("Pulquería", 87, (690, 20)))
    n1a = GraphNode(Sitio("Sambuca", 0, (350, 400)))
    n1b = GraphNode(Sitio("Convivio casa Alan", 98, (170, 400)))

    nodos_destino = {n00, n01, n02, n03, n04, n05, n06, n07, n08, n09, n0a, n0c, n0d, n0e, n0f, n11, n12, n13, n14, n15,
                     n16, n17, n18, n19, n1a, n1b}

    grafo = Graph(fi, nodos_destino)

    grafo.add_edge(fi, n01, randint(0, 100))
    grafo.add_edge(fi, n0d, randint(0, 100))
    grafo.add_edge(n0d, n06, randint(0, 100))
    grafo.add_edge(n06, n05, randint(0, 100))
    grafo.add_edge(n06, n14, randint(0, 100))
    grafo.add_edge(n05, n19, randint(0, 100))
    grafo.add_edge(n01, n1a, randint(0, 100))
    grafo.add_edge(n01, n18, randint(0, 100))
    grafo.add_edge(n01, n14, randint(0, 100))
    grafo.add_edge(n18, n1a, randint(0, 100))
    grafo.add_edge(n18, n1b, randint(0, 100))
    grafo.add_edge(n18, n11, randint(0, 100))
    grafo.add_edge(n18, n17, randint(0, 100))
    grafo.add_edge(n1a, n12, randint(0, 100))
    grafo.add_edge(n14, n12, randint(0, 100))
    grafo.add_edge(n12, n16, randint(0, 100))
    grafo.add_edge(n12, n09, randint(0, 100))
    grafo.add_edge(n09, n03, randint(0, 100))
    grafo.add_edge(n09, n08, randint(0, 100))
    grafo.add_edge(n08, n0a, randint(0, 100))
    grafo.add_edge(n09, n07, randint(0, 100))
    grafo.add_edge(n09, n0c, randint(0, 100))
    grafo.add_edge(n03, n08, randint(0, 100))
    grafo.add_edge(n0c, n07, randint(0, 100))
    grafo.add_edge(n0c, n13, randint(0, 100))
    grafo.add_edge(n0c, n04, randint(0, 100))
    grafo.add_edge(n13, n00, randint(0, 100))
    grafo.add_edge(n07, n0f, randint(0, 100))
    grafo.add_edge(n07, n02, randint(0, 100))
    grafo.add_edge(n02, n0e, randint(0, 100))
    grafo.add_edge(n0e, n04, randint(0, 100))
    grafo.add_edge(n04, n15, randint(0, 100))

    print("".center(term_width, '-'))
    print(f"|{'Facultad de Ingeniería - UNAM'.center(term_width - 2, ' ')}|")
    print(f"|{'Inteligencia Artificial - Semestre 2024-2'.center(term_width - 2, ' ')}|")
    print(f"|{'Práctica 3 - Búsqueda informada y búsqueda local'.center(term_width - 2, ' ')}|")
    print("".center(term_width, '-'))

    print(f"Grafo:")
    grafo.print_graph()

    print("[ Actividad 1 ]".center(term_width, '='))
    interes = 100
    resultados: list[tuple[list[GraphNode], int]] = []
    for des in nodos_destino:
        recorrido = grafo.a_star(fi, des, heuristic_p03)
        if len(recorrido) >= 3:
            resultados.append((recorrido, interes))
        interes = 100

    for datos in sorted(resultados, key=lambda x: x[1]):
        print(f"Interes restante: {datos[1]:.2f} Recorrido: {datos[0]}")

    mejor = max(resultados, key=lambda x: x[1])
    print("\033[92m", end="")
    print(f"Mejor recorrido: {mejor[0]}")
    print(f"Interés restante: {mejor[1]}")
    print("\033[0m", end="")

    print("[ Actividad 2 ]".center(term_width, '='))
    for node in nodos_destino:
        resultado: GraphNode[Sitio] = grafo.ascenso_colina(node, lambda n: n.value.interes)
        print(f"Mínimo partiendo de {node.value.nombre}: {resultado.value.nombre}")