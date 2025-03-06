## Project Description 


## Data augmentation 

We choose an unusual method to test data augmentation. We will test several (two or three) architectures on 100% of the data, 80%, 60%, 40%, and 20%. Those tests will be treated as points of reference. Then, we will create models based on datasets where the number of observations is equal to the full original dataset, but we generate a fraction of that observation using different argumentation techniques. 

Thanks to designing the experiment this way, we achieved two things:
 - we have clear points of reference that allow us to meaningfully measure model performance and thus different methods of augmentation
 - if we increase the number of observations, we would have to train a model for longer, requiring even more processing power, which we lack. The constant size of datasets results in a predictable training time, an important characteristic with such a short project time span. We will be able to test (and, because of it, learn) more in this project. Final performance is not as important as it is a research project.

 The convolutional neural network created in this experiment will be fairly simple. We plan to run each configuration multiple times to get an estimate of variance, and there are a lot of hyperparameters. In order to complete this section on time, we must make some compromises.