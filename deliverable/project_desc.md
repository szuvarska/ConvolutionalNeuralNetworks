## Project Overview
This is the report for the first project in the Deep Learning course in Data Science Masters at the Faculty of Mathematics and Information Science. It aims to introduce us to convolutional neural networks and develop intuitions related to them. We were informed that this work's goal is purely educational and research-focused, so we aim not to create the best recognition model but to learn as much as possible.

## Project Details 

The exact topic of this project is "Image classification with convolutional neural networks". We have been given the data set "CINIC-10", which we must use for this project. We decided to use PyTorch as a deep learning framework. Things that have to be included in this report:

1. Test and compare different network architectures (at least one should be a convolutional neural network).
    - Investigate the influence of the following hyper-parameter change on obtained results:
        - At least 2 hyper-parameters related to training process.
        - At least 2 hyper-parameters related to regularization.
    - Investigate the influence of at least x data augmentation techniques from the following groups:
        - Standard operations (x = 3).
        - More advanced data augmentation techniques like cutmix, cutout, or AutoAugment (x = 1).
2. Implement one method dedicated to few-shot learning.
    - Reduce the size of the training set and compare the obtained results with those trained on the entire dataset (from the previous point).
3. Consider the application of ensemble (hard/soft voting, stacking).

## Dataset Description
[CINIC-10](https://www.kaggle.com/datasets/mengcius/cinic10) is a dataset for image classification, and it was designed to serve as a bridge between the widely used CIFAR-10 dataset and the significantly larger ImageNet dataset. Images are split into three subsets: training, validation, and testing. 

Like CIFAR-10, CINIC-10 includes 10 different classes, which are categories into which the images are grouped. These classes are:

1. **Airplane**
2. **Automobile**
3. **Bird**
4. **Cat**
5. **Deer**
6. **Dog**
7. **Frog**
8. **Horse**
9. **Ship**
10. **Truck**

Each class has 9,000 images in each of the three subsets (training, validation, and testing), so in total, there are 90,000 images per subset and 270,000 images in the whole dataset. The images in CINIC-10 come from two different sources: 60,000 images are from CIFAR-10, and the remaining 210,000 images are from ImageNet, but they have been resized to 32x32 pixels to match the size of the CIFAR-10 images. 

CINIC-10 is particularly useful for training and testing image classification models that require more data than CIFAR-10 but are not yet ready for the computational complexity of the full ImageNet dataset, making it an ideal choice for learning deep learning models without extensive computing power.

After some research, we found that the best model trained on this data is `VIT-L/16`. It was introduced in this [paper](http://arxiv.org/pdf/2305.03238v6), and it achieved 95.5% accuracy. Other models worth mentioning are `DenseNet-121` with 91.26%, `ResNEt-18` with 90.27% and `VGG-16` with 87.77% accuracy.

**Idk if we want it in final report**
but i found [website](https://paperswithcode.com/dataset/cinic-10) with papers about this dataset so we might take a look at it at some point 

## Plan of experiments  


**I guess this part will not be included in the final report and if so it has to be rewritten**
1. Create CNN implementation in PyTorch and check parameters (time of training, accuracy, loss function)
2. Tweak this. For example, add one more convolutional layer, a different learning rate, max pooling instead of average polling, or the other way around, and maybe tweak a number of neurons. Play with the blueprint created in the first experiment. Compare it to ready architecture from the internet and ready model. With ready architecture, use all random weights and kernels and another variant with random weights but not kernels. 
3. Apply regularization to weights in the fully connected layer, check results, and if possible, apply regularization to kernel functions. Try with l1 and l2, maybe elastic net. Search for different methods.
4. Data augmentation as described below
    - rotations
    - flipping
    - cropping
    - brightness adjustment 
    - color jitter
    - Gaussian noise
    - some combination of those (a lot of experiments)
    - cutoff
    - maybe AutoAugment
5. Few-shot learning something we don't know yet
6. Probably stacking, but it seems easy, so we can implement soft voting as well. One thing to remember is that those models should vary fundamentally. If those models extract the same information from the image, then combining it does not help 


## Description of particular experiments

### General experiments 

### Hyper-parameters

### Data augmentation 

We choose an unusual method to test data augmentation. We will test several (two or three) architectures on 100% of the data, 80%, 60%, 40%, and 20%. Those tests will be treated as points of reference. Then, we will create models based on datasets where the number of observations is equal to the full original dataset, but we generate a fraction of that observation using different argumentation techniques. 

Thanks to designing the experiment this way, we achieved two things:

 - we have clear points of reference that allow us to meaningfully measure model performance and thus different methods of augmentation
 - if we increase the number of observations, we would have to train a model for longer, requiring even more processing power, which we lack. The constant size of datasets results in a predictable training time, an important characteristic with such a short project time span. We will be able to test (and, because of it, learn) more in this project. Final performance is not as important as it is a research project.

 The convolutional neural network created in this experiment will be fairly simple. We plan to run each configuration multiple times to get an estimate of variance, and there are a lot of hyper-parameters. In order to complete this section on time, we must make some compromises.

 ### Few shot learning

 ### Ensemble 
