Dimo update:
Did pretraining, the graph looks a bit weird, there is a drop from 2k to 4k.
Possible solution: increase the steps until you reach the full potential (unclear how?). Maybe decrease the batch size? Maybe 2k isn't a subset of 4k?
To improve the comparison it can make sense to train a plain classifier for fine-tuning to see the effect of pre-training. 
Don't use the full CIFAR dataset for fine-tuning to see the results of pre-training, use a dataset that has less features.

Leonid update:
Adapted the DINO implementation. Currently takes 48 seconds for one epoch. The full run-time estimation is concerning. If the training takes more than a day then it is better to stop, but 8 hours is still reasonable. Maybe tweak the GPU, access the cluster? Analyze the trade-off between the number of crops and the number of epochs in the paper
Rule of thumb: more epochs for a smaller dataset or a smaller batch size to keep the gradient step the same.
For the trend analysis you can only check 1%, 5%, 10%, 25%, 50%, 100% of the original dataset. No need to use the same step if the dependence is clear.

Makar update:
Tried to do something, nothing works yet.

Maksim update:
The loss plot acts weird. If it goes up, it can still be meaningful. 100 epochs is not nearly enough, need approximately about 500. It is common to do something like knn every 5 epochs to see the progress.

Fine-tuning evaluation: can choose between linear probing and PEFT. Linear probing demands less compute. Justify the choice in your project.

Need to schedule a meeting with Jan, wait for an update on mattermost.