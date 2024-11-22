import random
import time
from typing import Generator

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from parameters import Weights, SimulationResult
from util import list_to_string


class ElevatorSaga:
    def __init__(self) -> None:
        self.url = "https://play.elevatorsaga.com/"
        self.driver = webdriver.Firefox()
        self.driver.get(self.url)
        self._init_buttons()

    def _init_buttons(self):
        self.save_button = self.driver.find_element(value="button_save")
        self.apply_button = self.driver.find_element(value="button_apply")

        for _ in range(7):
            speed_up_button = self.driver.find_element(
                by=By.CLASS_NAME, value="fa-plus-square"
            )
            speed_up_button.click()
            time.sleep(0.1)

    def initialize_net(self):
        ActionChains(self.driver).key_down(Keys.CONTROL).key_down("a").key_up(
            Keys.CONTROL
        ).key_up("a").perform()
        ActionChains(self.driver).key_down(Keys.DELETE).key_up(Keys.DELETE).perform()

        weights = self._init_weights()
        self._insert_code(weights)

    def change_weights(self, weights: Weights) -> None:
        self._insert_code(weights)

    def _insert_code(self, weights: Weights) -> None:
        code_block = self.driver.find_element(by=By.CLASS_NAME, value="CodeMirror")
        code_block.click()

        new_code = (
            """{
                    init: function(elevators, floors) {
                        var elevator = elevators[0];
                        const weights = {
                            INPUTTOHIDDEN1,
                            HIDDEN1BIAS,
                            HIDDEN1TOHIDDEN2,
                            HIDDEN2BIAS,
                            HIDDEN2TOOUTPUT,
                            OUTPUTBIAS, 
                        };
                
                        elevator.on("idle", () => {
                            const input = this.getInput(elevator, floors);
                            const output = this.forwardPass(input, weights);
                            console.log(output[0]);
                            if (output[0] > 0.5) {
                                elevator.goToFloor(Math.min(floors.length - 1, elevator.currentFloor() + 1));
                            } else if (output[0] <= 0.5) {
                                elevator.goToFloor(Math.max(0, elevator.currentFloor() - 1));
                            }
                            else {elevator.goToFloor(elevator.currentFloor())};
                        });
                    },
                    update: function(dt, elevators, floors) {},
                    
                    getInput: function (elevator, floors) {
                        const inputs = [];
                        inputs.push(elevator.currentFloor());
                        inputs.push(elevator.maxPassengerCount());
                        inputs.push(elevator.loadFactor());
                        inputs.push(this.directionMap[elevator.destinationDirection()] || 0);
                        inputs.push(Number(elevator.goingUpIndicator()));
                        inputs.push(Number(elevator.goingDownIndicator()));
                        pressedFloors = elevator.getPressedFloors();
                        for (const floor of floors) {
                            inputs.push(pressedFloors.includes(floor.level) ? 1 : 0);
                        };
                        for (let i=floors.length; i < 7; i++) {
                            inputs.push(0);
                        };
                        
                        return inputs
                    },
                        
                    sigmoid: function (x) {
                        return 1 / (1 + Math.exp(-x));
                    },
                
                    forwardPass: function (input, weights) {
                        const dotProduct = (a, b) => {
                            const result = [];
                            for (let i = 0; i < b.length; i++) {
                                let sum = 0;
                                for (let j = 0; j < a.length; j++) {
                                    sum += a[j] * b[i][j];
                                }
                                result.push(sum);
                            }
                            return result;
                        };
                
                        const addBias = (layer, bias) => {
                            return layer.map((val, i) => {
                                return val + bias[i];
                            });
                        };
                
                        const applyActivation = (layer) => {
                            return layer.map(this.sigmoid);
                        };
                
                        let hidden1 = dotProduct(input, weights.inputToHidden1);
                        hidden1 = addBias(hidden1, weights.hidden1Bias);
                        // hidden1 = applyActivation(hidden1);
                        
                        let hidden2 = dotProduct(hidden1, weights.hidden1ToHidden2);
                        hidden2 = addBias(hidden2, weights.hidden2Bias);
                        // hidden2 = applyActivation(hidden2);
                        
                        let output = dotProduct(hidden2, weights.hidden2ToOutput);
                        output = addBias(output, weights.outputBias);
                        output = applyActivation(output);
                
                        return output;
                    },
                
                    numInputs: 13,
                    numHidden1: 10,
                    numHidden2: 10,
                    numOutputs: 2,
                    directionMap: { up: 1, down: -1 },
                }
                
                """.replace(
                "INPUTTOHIDDEN1",
                "inputToHidden1: " + list_to_string(weights.input_to_hidden_1),
            )
            .replace(
                "HIDDEN1BIAS",
                "hidden1Bias: " + list_to_string(weights.hidden_1_bias),
            )
            .replace(
                "HIDDEN1TOHIDDEN2",
                "hidden1ToHidden2: " + list_to_string(weights.hidden_1_to_hidden_2),
            )
            .replace(
                "HIDDEN2BIAS",
                "hidden2Bias: " + list_to_string(weights.hidden_2_bias),
            )
            .replace(
                "HIDDEN2TOOUTPUT",
                "hidden2ToOutput: " + list_to_string(weights.hidden_2_to_output),
            )
            .replace("OUTPUTBIAS", "outputBias: " + list_to_string(weights.output_bias))
        )
        self.driver.execute_script(
            "arguments[0].CodeMirror.setValue(arguments[1]);", code_block, new_code
        )

    def run_simulation(self) -> None:
        self.save_button.click()
        self.apply_button.click()
        time.sleep(1)

    def get_result(self) -> SimulationResult:
        results_fields = [
            "transportedcounter",
            "elapsedtime",
            "transportedpersec",
            "avgwaittime",
            "maxwaittime",
            "movecount",
        ]

        results = []
        for field in results_fields:
            element = self.driver.find_element(by=By.CLASS_NAME, value=field)
            results.append(float(element.text.strip("s")))

        return SimulationResult(*results)

    def _init_weights(self) -> Weights:
        # fmt: off
        hidden_1_bias = [random.uniform(-1, 1) for _ in range(10)]
        hidden_1_to_hidden_2 = [[random.uniform(-1, 1) for _ in range(10)] for _ in range(10)]
        hidden_2_bias = [random.uniform(-1, 1) for _ in range(10)]
        hidden_2_to_output = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(10)]
        input_to_hidden_1 = [[random.uniform(-1, 1) for _ in range(10)] for _ in range(13)]
        output_bias = [[random.uniform(-1, 1)] for _ in range(2)]
        # fmt: on

        return Weights(
            hidden_1_bias,
            hidden_1_to_hidden_2,
            hidden_2_bias,
            hidden_2_to_output,
            input_to_hidden_1,
            output_bias,
        )


def string_generator(text: str) -> Generator[str, None, None]:
    for letter in text:
        yield letter


if __name__ == "__main__":
    elevator_saga = ElevatorSaga()
    elevator_saga.initialize_net()
    elevator_saga.run_simulation()
    elevator_saga.get_result()
