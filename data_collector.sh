#!/bin/sh

python3 training_main.py --save-to models/network1/FinalTrained -c --two_class
python3 training_main.py --save-to models/newtwork2/FinalTrained -c --two_class
python3 training_main.py --save-to models/newtwork3/FinalTrained -c --two_class
python3 training_main.py --save-to models/newtwork4/FinalTrained -c --two_class
python3 training_main.py --save-to models/newtwork5/FinalTrained -c --two_class
python3 training_main.py --save-to models/newtwork5/FinalTrainedRegularized -c --two_class --L2 1

