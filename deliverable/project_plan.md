## Project Description 


## Data augmentation 

We choose an unusual method to test data augmentation. We will test several (two or three) architectures on 100% of the data, 80%, 60%, 40%, and 20%. Those tests will be treated as points of reference. Then, we will create models based on datasets where the number of observations is equal to the full original dataset, but we generate a fraction of that observation using different argumentation techniques. 

Thanks to designing the experiment this way, we achieved two things:
 - we have clear points of reference that allow us to meaningfully measure model performance and thus different methods of augmetnation
 - if we would increase number of observation we would have to train model for longer, and it would require even move processing power which we lack. Constant size of datasets results in predictable time of training, an important characteristic with such short time span of project. We will be able to test (and bacause of it learn) more. In this project final performance in not that important as it is a reaserch project.

 The convolutional neural network created in this experiments will be fairly simple. We plan to run each configuration multiple time to get an estimate on variance and there is a lot of hyper parameters. In order to complete this section on time we have to make some compromises.