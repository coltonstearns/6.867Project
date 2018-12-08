#!/bin/sh



#extra testing for model 1
#python3 training_main.py --save-to models/network1/FinalTrainedPlainTest -c --two_class --per_class --test -l models/network1/FinalTrained
#python3 training_main.py --save-to models/network1/FinalTrainedPrior -c --two_class --per_class --test --prior  -l models/network1/FinalTrained
#python3 training_main.py --save-to models/network1/FinalTrainedCRF -c --two_class --per_class --test --use_crf  -l models/network1/FinalTrained 

#Extra testing for model 2
#python3 training_main.py --save-to models/network2/FinalTrainedPlainTest -c --two_class --per_class --test -l models/network2/FinalTrained 
#python3 training_main.py --save-to models/network2/FinalTrainedPrior -c --two_class --per_class --test --prior -l models/network2/FinalTrained 
#python3 training_main.py --save-to models/network2/FinalTrainedCRF -c --two_class --per_class --test --use_crf -l models/network2/FinalTrained 

#Extra testing for model 3
#python3 training_main.py --save-to models/network3/FinalTrainedPlainTest -c --two_class --per_class --test -l models/network3/FinalTrained 
python3 training_main.py --save-to models/network3/FinalTrainedPrior -c --two_class --per_class --test --prior -l models/network3/FinalTrained 
#python3 training_main.py --save-to models/network3/FinalTrainedCRF -c --two_class --per_class --test --use_crf -l models/network3/FinalTrained 

#Extra testing for model 4
#python3 training_main.py --save-to models/network4/FinalTrainedPlainTest -c --two_class --per_class --test -l models/network4/FinalTrained 
python3 training_main.py --save-to models/network4/FinalTrainedPrior -c --two_class --per_class --test --prior -l models/network4/FinalTrained 
#python3 training_main.py --save-to models/network4/FinalTrainedCRF -c --two_class --per_class --test --use_crf -l models/network4/FinalTrained 

# Model 5 training
#python3 training_main.py --save-to models/network5/FinalTrained -c --two_class --per_class

# Model 5 Testing
#python3 training_main.py --save-to models/network5/FinalTrainedPlainTest -c --two_class --per_class --test -l models/network5/FinalTrained
#python3 training_main.py --save-to models/network5/FinalTrainedPrior -c --two_class --per_class --test --prior  -l models/network5/FinalTrained
#python3 training_main.py --save-to models/network5/FinalTrainedCRF -c --two_class --per_class --test --use_crf  -l models/network5/FinalTrained 

#Regularized Model 5 Training
python3 training_main.py --save-to models/network5/FinalTrainedReg -c --two_class --L2 1 --per_class

#Regularized Model 5 testing
python3 training_main.py --save-to models/network1/FinalTrainedRegPlainTest -c --two_class --per_class --test -l models/network5/FinalTrainedReg
python3 training_main.py --save-to models/network1/FinalTrainedRegPrior -c --two_class --per_class --test --prior  -l models/network1/FinalTrainedReg
#python3 training_main.py --save-to models/network5/FinalTrainedRegCRF -c --two_class --per_class --test --use_crf  -l models/network1/FinalTrainedReg 

