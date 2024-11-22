from dataclasses import dataclass

# hidden1Bias: Array(10)                 10
# hidden1ToHidden2: Array(10) [ (10) ]  100
# hidden2Bias: Array(10)                 10
# hidden2ToOutput: Array(10) [ (2) ]     20
# inputToHidden1: Array(13) [ (10) ]    130
# outputBias: Array(2) [ (1) ]            2
# -------------------------------------------
#                                       272


@dataclass
class Weights:
    hidden_1_bias: list[float]
    hidden_1_to_hidden_2: list[list[float]]
    hidden_2_bias: list[float]
    hidden_2_to_output: list[list[float]]
    input_to_hidden_1: list[list[float]]
    output_bias: list[list[float]]


@dataclass
class SimulationResult:
    transported_people: float
    elapsed_time: float
    transported_per_second: float
    average_waiting_time: float
    max_waiting_time: float
    move_count: float

    def get_fitness(self) -> float:
        return 1