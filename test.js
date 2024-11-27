{
    init: function(elevators, floors) {

        elevators.forEach((elevator, i) => {
            const weights = {
                INPUTTOHIDDEN1,
                HIDDEN1BIAS,
                HIDDEN1TOHIDDEN2,
                HIDDEN2BIAS,
                HIDDEN2TOOUTPUT,
                OUTPUTBIAS, 
            };
    
            elevator.on("idle", () => {
                const input = this.getInput(elevators, floors);
                const output = this.forwardPass(input, weights);
                elevatorNo = i;
                console.log(output[elevatorNo*2], output[elevatorNo*2+1]);
                if (output[elevatorNo*2] > output[elevatorNo*2+1]) {
                    elevator.goToFloor(Math.min(floors.length - 1, elevator.currentFloor() + 1));
                } else {
                    elevator.goToFloor(Math.max(0, elevator.currentFloor() - 1));
                }
            });
        });
    },
    update: function(dt, elevators, floors) {},
    
    getInput: function (elevators, floors) {
        const inputs = [];
        elevators.forEach((elevator, i) => {
            if (elevators.count > i) {
                inputs.push(elevators[i].currentFloor());
                inputs.push(this.directionMap[elevators[i].destinationDirection()] || 0);
                inputs.push(elevators[i].loadFactor());
                inputs.push(elevators[i].maxPassengerCount());
                inputs.push(elevators[i].goingUpIndicator());
                inputs.push(elevators[i].goingDownIndicator());
            } else {
                for(let j = 0; j < 6; j++){
                    inputs.push(-1);
                }
            }
        });
        pressedFloors = elevator.getPressedFloors();
        for (const floor of floors) {
            inputs.push(pressedFloors.includes(floor.level) ? 1 : 0);
        };
        for (let i=floors.length; i < 7; i++) {
            inputs.push(-1);
        };
        inputs.push(floors.count);
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
        
        let hidden2 = dotProduct(hidden1, weights.hidden1ToHidden2);
        hidden2 = addBias(hidden2, weights.hidden2Bias);
        
        let output = dotProduct(hidden2, weights.hidden2ToOutput);
        output = addBias(output, weights.outputBias);
        output = applyActivation(output);

        return output;
    },

    numInputs: 32,
    numHidden1: 64,
    numHidden2: 16,
    numOutputs: 8,
    directionMap: { up: 1, down: -1 },
}