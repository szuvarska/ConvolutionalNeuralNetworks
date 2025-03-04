## Project Description 


## Data augmentation 

We choose an unusual method to test data augmentation. We will test several (two or three) architectures on 100% of the data, 80%, 60%, 40%, and 20%. Those tests will be treated as points of reference. Then, we will create models based on datasets where the number of observations is equal to the full original dataset, but we generate a fraction of that observation using different argumentation techniques. 

Thanks to designing the experiment this way, we achieved two things:
 - we have clear points of reference that allow us to meaningfully measure model performance in comparison to