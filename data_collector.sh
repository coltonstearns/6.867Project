#!/bin/sh

python3 training_main.py --save-to models/network1/FinalTrained -c --two_class --per_class
python3 training_main.py --save-to models/network2/FinalTrained -c --two_class --per_class
python3 training_main.py --save-to models/network3/FinalTrained -c --two_class --per_class
python3 training_main.py --save-to models/network4/FinalTrained -c --two_class --per_class
python3 training_main.py --save-to models/network5/FinalTrained -c --two_class --per_class
python3 training_main.py --save-to models/network5/FinalTrainedRegularized -c --two_class --L2 1 --per_class

