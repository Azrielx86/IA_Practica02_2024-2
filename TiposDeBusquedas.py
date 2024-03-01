import math
from typing import TypeVar, Optional, Callable

T = TypeVar("T")


class GraphNode:
    def __init__(self, value: T) -> None:
        """
        :param value: Data to store in the node
        """
        self.__value: T = value
        self.__neighbours: dict[GraphNode, int] = {}
        self.__visited: bool = False

    @property
    def visited(self):
        return

    @visited.setter
    def visited(self, value):
        self.__visited = value

    @property
    def neighbours(self):
        return self.__neighbours

    @neighbours.setter
    def neighbours(self, value):
        self.__neighbours = value

    @property
    def value(self) -> T:
        return self.__value

    @value.setter
    def value(self, value: T) -> None:
        self.__value = value

    def add_neighbour(self, node: "GraphNode", dist: int):
        if node not in self.__neighbours.keys():
            self.__neighbours[node] = dist
        else:
            print(f"Node {node} already in {self} neighbours list")

    def __str__(self) -> str:
        return str(self.__value)

    def __repr__(self) -> str:
        return self.__str__()


class Graph:
    def __init__(self) -> None:
        self.__root: GraphNode | None = None
        self.__vertex: set[GraphNode] = set()

    @property
    def vertex(self):
        return

    @property
    def root(self) -> GraphNode:
        return self.__root

    @root.setter
    def root(self, value: GraphNode) -> None:
        self.__root = value

    def add_edge(self, node1: GraphNode, node2: GraphNode, weight: int) -> None:
        node1.add_neighbour(node2, weight)
        node2.add_neighbour(node1, weight)

        if node1 not in self.__vertex:
            self.__vertex.add(node1)

        if node2 not in self.__vertex:
            self.__vertex.add(node2)

    def breadth_first_search(self, stop_condition: Optional[Callable]) -> list[GraphNode]:
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
            node.visited = True
            result.append(node)

            pending = [n for n in node.neighbours if n is not n.visited]

            for n in pending:
                queue.append(n)

            if stop_condition and stop_condition(node.value):
                break

        for n in self.__vertex:
            n.visited = False

        return result

    def depth_first_search_iterative(self, stop_condition: Optional[Callable]):
        """
        Preorder traverse, iterative solution.
            1. Visit the root
            2. Traverse the left subtree, i.e., call Preorder(left-subtree)
            3. Traverse the right subtree, i.e., call Preorder(right-subtree)
        :param stop_condition: Condition to search a node and stop the algorithm
        :return: A list with the traveled nodes
        """
        stack: list = []
        result: list = []
        node = self.__root

        while stack or node:
            if node:
                result.append(node)

                if right := node.rightChild:
                    stack.append(right)

                node = node.leftChild
            else:
                node = stack.pop()

            if stop_condition and stop_condition(node.value):
                result.append(node)
                break

        return result


def generateTree(nodeData: T, level: int, maxLevel: int, weight: int = 0) -> GraphNode:
    """
    Generates a tree with the condition:
        - Left node: :math:`x-1`
        - Right node: :math:`\\sqrt{x}`
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

    left_child = generateTree(left, level + 1, maxLevel)
    right_child = generateTree(right, level + 1, maxLevel)

    node.add_neighbour(left_child, weight)
    node.add_neighbour(right_child, weight)

    return node


if __name__ == '__main__':
    tree = Graph()
    tree.root = generateTree(25, 0, 5)
    print(f"BFS travel: {tree.breadth_first_search(lambda x: x == 30)}")
    # print(f"DFS travel: {tree.depthFirstSearch(lambda x: x == 30)}")
    print("A")
