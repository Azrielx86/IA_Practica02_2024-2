import random

MIN_VAL = 0
MAX_VAL = 9

atributos = ["Costo", "Riesgo", "Aventura", "Distancia", "Atracciones", "Conocido", "Duración", "Paisaje"]
modelo = [4, 5, 9, 8, 7, 2, 6, 9]
largo = len(modelo)
num = 5
pressure = 3
mutation_chance = 0.45
mostrar_cruces = False

try:
    assert len(atributos) == len(modelo)
except AssertionError:
    print("Atributos != Modelo")
    exit(1)

print(f"Modelo: {modelo}")

for atributo, valor in zip(atributos, modelo):
    print(f"{atributo} = {valor}")


def individual(min_val: int, max_val: int) -> list[int]:
    return [random.randint(min_val, max_val) for _ in range(largo)]


def crearPoblacion() -> list[list[int]]:
    return [individual(MIN_VAL, MAX_VAL) for _ in range(num)]


def calcularFitness(individual: list):
    fitness = 0
    for i in range(len(individual)):
        if individual[i] == modelo[i]:
            fitness += 1

    return fitness


def create_child(parent1: list, parent2: list, point1: int, point2: int) -> list:
    assert len(parent1) == len(parent2)
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

    # Esto es meramente ilustrativo
    if mostrar_cruces:
        sel_p1 = ['-' for i in range(len(parent1))]
        sel_p2 = ['-' for i in range(len(parent1))]
        sel_p1[:point1] = parent1[:point1]
        sel_p2[point1:point2] = parent2[point1:point2]
        sel_p1[point2:] = parent1[point2:]

        print(f"Genes por cruzar para el primer hijo")
        print(''.join([*map(lambda c: str(c), sel_p1)]))
        print(''.join([*map(lambda c: str(c), sel_p2)]))

        sel_p1 = ['-' for i in range(len(parent1))]
        sel_p2 = ['-' for i in range(len(parent1))]
        sel_p1[:point1] = parent2[:point1]
        sel_p2[point1:point2] = parent1[point1:point2]
        sel_p1[point2:] = parent2[point2:]

        print(f"Genes por cruzar para el segundo hijo")
        print(''.join([*map(lambda c: str(c), sel_p2)]))
        print(''.join([*map(lambda c: str(c), sel_p1)]))

    return [child1, child2]


def selection_and_reproduction(population: list[list]):
    puntuados = [(calcularFitness(i), i) for i in population]
    puntuados = [i[1] for i in sorted(puntuados, key=lambda x: x[0])]  # ordenar por fitness
    population = puntuados

    selected = puntuados[(len(puntuados) - pressure):]

    for i in range(len(population) - pressure):
        punto1 = random.randint(1, largo - 1)
        punto2 = random.randint(punto1, largo)
        padres = random.sample(selected, 2)
        childs = create_child(padres[0], padres[1], punto1, punto2)

        if mostrar_cruces:
            print('Cruzamiento'.center(80, '-'))
            print(f"Padres: {padres}")
            print(f"Puntos de cruzamiento: [{punto1}, {punto2}]")
            print(f"Hijos obtenidos: {childs}")

        # Agrega los dos hijos a la población
        population[i: i + 2] = childs[0:2]

    return population


def mutation(population):
    for i in range(len(population) - pressure):
        if random.random() <= mutation_chance:
            punto = random.randint(0, largo - 1)
            nuevo_valor = random.randint(MIN_VAL, MAX_VAL)

            while nuevo_valor == population[i][punto]:
                nuevo_valor = random.randint(MIN_VAL, MAX_VAL)

            population[i][punto] = nuevo_valor

    return population


encontrados: dict[int, int] = {10: 0, 50: 0, 100: 0}
for prueba in range(1):
    for iteraciones in [10, 50, 100]:
        print(f"{iteraciones} iteraciones".center(80, '='))
        population = crearPoblacion()
        print(f"Poblacion Inicial {population}")
        for i in range(iteraciones):
            population = selection_and_reproduction(population)
            population = mutation(population)
        print("\33[2K", end="")
        print(f"Poblacion Final   {population}")
        if modelo in population:
            print(f"Modelo encontrado con {iteraciones} iteraciones.")
            encontrados[iteraciones] += 1
print(f"Modelo encontrado {encontrados}")
