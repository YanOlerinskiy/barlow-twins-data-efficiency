Q (Yan): How 

Q: How does the grading work? Are you involved? There is a part for the process
A: They are not directly involved. We think the process is pretty good, everyone is proactive when it comes to asking questions

Q (Max, Dimo): We wrote a protocol of shared linear probing setup, the protocol requires to pool the features with a CLS token, which is not part I-JEPA objective, so the result is garbage. We get the same result by using global average pooling. Is it fine to have 2 baseline lines 
A: 

Q: In the protocol we use the best accuracy over 100 epochs as the best score. Is it representative? If it is super volatile the score will be useless. 
A: Are you tuning on the test set? (Yes). Is that a problem given Yan's paper? We don't like it, better pick a validation set and do it on it.
Q: Maybe it's worth to just take the result of the last epoch? 
A: Maybe it's easier if you do that, yes 

Q (Dimo): Bayesian optimization. I should decide on the number of epochs for each dataset, is it fine to find a set of good hyperparams for a small dataset and use the same for the bigger datasets? Do I fix the epochs or do I have to do something else?
A: For every dataset try and give it the best circumstances. You can fix the epochs if you see it converges, but the effect of changing the epochs can be investigated on its own. It's important to not change multiple things at the same time though.

Q (Makar): Q about dataset for pre-training. I looked into improving augs, i.e. adding multi-crop, but because our images are small we can't do much. I was thinking to switch from 64x64 to 128x128 (resized imagenet for example) and to increase the size of the patch from 8 to 16, this won't change the number of ViT parameters. This could be helpful for multi-crop. Is it worth it to look in this direction?
A: I don't think it's worth it changing everyone's datasets at this point.

Q (Makar): I got access to DAIC. How slow is it to execute jobs? What was your experience? 
A: Availability-wise it's fine. But do you want to spend ~2 days to set up the DAIC, make a container, transfer images? In some it's a financial decision if it's worth it to spend a few days getting the cluster to work. The worse GPUs on DAIC are usually more available. V100 should be fine for everyone. The worst wait time for a job is 2-3 days but there are no big conferences now. If you use DAIC, you should use uptainer, it's like docker but on DAIC.

Q (Max): In the fine-tuning protocol, there are 2 options: freeze the encorer, calculate once for the whole cifar, or you random horizontal flip which forces us to recalculate embedding - for me it's 0.5% or just noisy. Running LP in seconds is much nicer for me, should we use the horizontal flip at all?
A: Doesn't sound like the flip adds any value for you, but it may for others. The thing we care about is that LP is same for everyone, so why would we improve LP? The whole storyline is about pre-training anyway.

Q (Makar): Right now we are doing fine-tuning and looking at downstream acc. Is there a way to see what the model actually learned? DINO can learn depth if it has enough data, is there anything similar we could do ourselves? 
A: Classifying on tiny imagenet might help to see if you learned the classes correctly.

1. augmentations?
2. ⁠fix overtraining, have cifar-10 validation and stop training based on it 
3. ⁠where does the training fail right now? 