from dataclasses import dataclass

# hidden1Bias: Array(64)                 64
# hidden1ToHidden2: Array(64) [ (16) ]  1024
# hidden2Bias: Array(16)                  16
# hidden2ToOutput: Array(16) [ (8) ]     128
# inputToHidden1: Array(32) [ (64) ]   2056
# outputBias: Array(8)                    8
# -------------------------------------------
#                                       3296


@dataclass
class Weights:
    hidden_1_bias: list[float]
    hidden_1_to_hidden_2: list[list[float]]
    hidden_2_bias: list[float]
    hidden_2_to_output: list[list[float]]
    input_to_hidden_1: list[list[float]]
    output_bias: list[float]


@dataclass
class SimulationResult:
    transported_people: float
    elapsed_time: float
    transported_per_second: float
    average_waiting_time: float
    max_waiting_time: float
    move_count: float

    def get_fitness(self) -> float:
        return (
            # 0.6*self.transported_people*self.transported_people
            # + 0.3*(60 - self.elapsed_time)*(60 - self.elapsed_time)
            # # - self.average_waiting_time
            # - self.max_waiting_time
            + self.move_count*self.move_count
        )
