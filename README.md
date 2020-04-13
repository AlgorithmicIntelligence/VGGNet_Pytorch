Implement the code from the [Paper]("https://arxiv.org/pdf/1409.1556.pdf"), all the architecture and parameters are the same as paper.

Difference with the Original:
1. Train the model with 100 classes.
2. Initial the parameters with KAMING, but I also provide the method of initializer with VGG and XAVIER.
3. Batch size 32.

The train/test accuracy are shown as below:

<img src="https://github.com/AlgorithmicIntelligence/VGGNet_Pytorch/blob/master/README/TrainAccuracy.png" width="450"><img src="https://github.com/AlgorithmicIntelligence/VGGNet_Pytorch/blob/master/README/TestAccuracy.png" width="450">

The train/test loss are shown as below:

<img src="https://github.com/AlgorithmicIntelligence/VGGNet_Pytorch/blob/master/README/TrainLoss.png" width="450"><img src="https://github.com/AlgorithmicIntelligence/VGGNet_Pytorch/blob/master/README/TestLoss.png" width="450">

Conclusion:

From the result, it suffers from overfitting, and there are indiscriminate between model VGG_A to VGG_E. I think the most possibility is that I only trained the model with 100 classes, so the quantity of dataset is deficient for the "DEEP" model for learning.