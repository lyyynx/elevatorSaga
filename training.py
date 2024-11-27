import random

import pygad

from elevator import ElevatorSaga
from parameters import Weights


def elevator_fitness(
    ga_instance: pygad.GA,
    solution: list[float],
    solution_idx: int,
    elevators: list[ElevatorSaga],
) -> float:
    weights = Weights(
        hidden_1_bias=list(solution[:10]),
        hidden_1_to_hidden_2=[list(solution[i : i + 10]) for i in range(10, 101, 10)],
        hidden_2_bias=list(solution[110:120]),
        hidden_2_to_output=[list(solution[i : i + 10]) for i in range(120, 131, 10)],
        input_to_hidden_1=[list(solution[i : i + 13]) for i in range(140, 261, 13)],
        output_bias=[solution[270], solution[271]],
    )

    fitness = 0.0
    for elevator in elevators:
        elevator.change_weights(weights)

        elevator.run_simulation()

        results = elevator.get_result()
        run_fitness = results.get_fitness()
        fitness += run_fitness

    print(
        f"epoch {ga_instance.generations_completed}, genome {solution_idx}, fitness {fitness}"
    )

    return fitness


if __name__ == "__main__":
    num_generations = 5
    num_parents_mating = 3
    sol_per_pop = 20
    num_genes = 272  # adapt
    init_range_low = -0.2
    init_range_high = 0.2
    parent_selection_type = "sss"
    keep_parents = 2
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 10

    elevator_sagas = [ElevatorSaga(level) for level in range(1, 4)]

    # fmt: off
    initial_population = [-2.32249275, -1.96938159,  0.50145911,  0.08751319, -3.14268459,  2.08514786, -0.56908317,  0.92564142,  0.92654585, -1.09341395, -0.31213009,  0.03581084, -1.11364309,  1.06986031,  0.48846948,  0.29080645,  0.06548625,  0.82830806, 0.6153867 ,  1.4067387 ,  0.93003976,  0.05113321,  2.61076566,  1.52472164, -1.0488221 , -0.84297492, -1.56553772, -0.84087003, -0.12851088, -0.5248412 , 0.98377316,  0.1754821 ,  1.10849216, -1.01777731, -1.91157325,  0.14307692, -0.5505963 ,  0.45333473,  0.05362736, -2.56875494,  0.96541042, -0.66675259, -1.49571333, -0.34770246, -0.64706239, -1.91159475,  0.07048383,  0.63556265, -1.90424255,  1.30280064, -1.09732947,  0.66903469, -1.14921139, -0.87655518, -2.00599689,  0.10064395,  1.80119914,  1.0614125 ,  1.11946176, -0.55563574, -0.01324333,  0.74581389,  0.5290584 , -1.18967742,  0.64462099, -0.54535069, -3.56871221, -2.37251155,  0.94728701, -0.8182108 , -0.28375382, -0.02012795, -0.08711833, -1.32597791, -1.99857213, -0.36935045, -0.76532683, -0.39329775, -0.02896115, -0.42730561, -0.53908938,  0.0757077 ,  0.1046892 , -1.3493364 , 1.73188282,  0.81608924,  0.47522915, -0.58088293, -0.26323274,  0.1424595 , -0.9995681 , -0.24438881, -0.3702067 ,  0.31785129,  0.64818548, -0.27058326, 0.4889666 , -0.96685486, -0.02512428, -0.70358251, -0.9251099 ,  0.40636518, 2.11751109,  0.27162268, -0.19047037,  0.64951649,  0.74700514,  0.97177349, -0.70169095,  0.41953858, -1.44737666,  0.44145974, -1.71676854, -0.83131702, 0.61410587, -0.2114406 , -0.8812547 , -1.86133759, -1.84163977, -1.83232618, 0.9103876 ,  1.53923964,  0.03346677, -0.78778262, -0.1501397 ,  1.37804858, -0.6680179 , -0.6106982 , -0.77576728,  1.51160434,  1.29103256, -1.5933084 , -3.14423231,  0.01606291,  0.54406344, -2.36542937, -1.06376223,  0.76521981, -0.96815188,  0.67603997, -0.46973832, -0.09678887,  2.85901555, -1.46696256, -0.52312125,  0.64550194,  1.27910998,  1.11371   ,  0.54906541,  0.37468528, 0.39992562,  0.73049733,  0.35529495, -0.37694028, -0.99824495, -1.69336805, 0.17367618,  1.19512543, -0.02526356,  0.76557446, -1.46114093,  0.46366155, -0.40433647,  0.69327721, -1.00773175, -0.19130164,  0.73004875, -1.36568453, 0.03503851,  0.70167727,  1.8592169 , -1.17107838, -1.60574401,  0.41419017, 2.29699615,  0.76250817,  0.45281407,  0.84562714, -1.68499627, -0.8038872 , 1.6053297 , -0.4842053 ,  0.74593492,  0.07291336,  2.70100429, -0.08378328, 0.5455594 ,  1.70732471,  0.49192498, -0.98349132,  0.72703437, -1.71265808, -0.63808642, -1.96762165,  0.40900211,  0.50569574,  1.171466  , -1.16323929, 1.08610564, -0.61124616, -1.31112462, -0.97576865,  0.45875712,  0.64519049, -0.29041402, -0.90326994,  0.59919083,  0.9230056 , -0.62549097,  0.91360096, -0.1268743 ,  0.54409123, -2.06055628, -1.39603973,  0.72946183, -1.02667964, -0.22677905, -0.59204208,  0.15317854,  0.42358279,  1.29564222, -0.67512437, -1.02243462, -0.44727562,  0.51487552,  0.11223451, -0.02921038, -1.22353638, -1.77496736,  1.40734642,  0.3624676 ,  0.71955985, -0.56802265,  0.01194511, 0.89597291,  0.54949616, -0.80306335, -0.7036734 ,  0.79310233,  0.90543714, -0.54461506,  1.97996508, -0.74505456,  0.68085708,  0.99405603,  1.0510922 , 0.47941553,  0.3034525 ,  0.99266937,  0.24226885, -0.75557272,  0.61927484, -0.02091646, -1.01265232,  0.49089197, -0.36641116,  1.12718059,  2.22307399, 0.42874824,  0.17672262,  0.6614553 , -0.46403613,  1.08838801, -0.57947975, -0.24302637, -0.68980964,  0.7842066 ,  0.33303085,  0.81860484,  0.13343733, 0.29584886,  0.11455332]
    # fmt: on

    randomized_initial_population = [
        initial_population,
        *[
            [entry + (random.random() - 0.5) for entry in initial_population]
            for _ in range(sol_per_pop - 1)
        ],
    ]

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=lambda instance, solution, solution_idx: elevator_fitness(
            instance, solution, solution_idx, elevator_sagas
        ),
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        init_range_low=init_range_low,
        init_range_high=init_range_high,
        initial_population=randomized_initial_population,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
        save_solutions=True,
    )

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print(
        "Fitness value of the best solution = {solution_fitness}".format(
            solution_fitness=solution_fitness
        )
    )

    ga_instance.plot_fitness()
    ga_instance.plot_new_solution_rate()
