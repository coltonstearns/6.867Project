#!/bin/sh



#model 8 training
python3 training_main.py --save-to models/network8/FinalTrained -c --two_class --per_class
python3 training_main.py --save-to models/network8/FinalTrainedPlainTest -c --two_class --per_class --test -l models/network8/FinalTrained
python3 training_main.py --save-to models/network8/FinalTrainedPrior -c --two_class --per_class --test --prior  -l models/network8/FinalTrained

# Model 7 training
python3 training_main.py --save-to models/network7/FinalTrained -c --two_class --per_class
python3 training_main.py --save-to models/network7/FinalTrainedPlainTest -c --two_class --per_class --test -l models/network7/FinalTrained
python3 training_main.py --save-to models/network7/FinalTrainedPrior -c --two_class --per_class --test --prior  -l models/network7/FinalTrained

#Model 8 3-class
python3 training_main.py --save-to models/network8/FinalTrained3 -c --per_class
python3 training_main.py --save-to models/network8/FinalTrained3PlainTest -c --per_class --test -l models/network8/FinalTrained
python3 training_main.py --save-to models/network8/FinalTrained3Prior -c --per_class --test --prior  -l models/network8/FinalTrained

#Model 7 3-class
python3 training_main.py --save-to models/network7/FinalTrained3 -c --per_class
python3 training_main.py --save-to models/network7/FinalTrained3PlainTest -c --per_class --test -l models/network7/FinalTrained
python3 training_main.py --save-to models/network7/FinalTrained3Prior -c --per_class --test --prior  -l models/network7/FinalTrained
