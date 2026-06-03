have a chair, update on progress, take notes 

how to decide on a topic? 
share a data loader (determines a dataset for specific sizes), share datasets of specific sizes to compare better 
decide on a dataset 
architecture? model? pre-training or fine-tuning (can be any, the goal so to improve data efficiency)? 
if we need access to the cluster, contact supervisors and jan and we can figure it out, but it’s not necessary. better to start with local computations 
first milestone: get to a learning curve (accuracy from data set size) 

to improve the curve, we need to do 2 things: 
1) pre-train (self-supervised, generalistic dataset, low-scale, otherwise too much compute required)
2) fine-tune (downstread task adaptation, peft is an example of improvement) 

read paper “masked autoencoders are scalable vision learners”
datasets: mnist, cifar-10, imagenet, fashion mnist
models: dino, byol, mae, moco, jepa, simclr
architectures: vit, cnn, ConvNeXt (vit inspired), compact transformer 

general advice for research questions:
make specific as possible, but not yes/no (“is <something> better” is not good, “how is <something> better” is better)

reach out in mattermost if any questions, ad hoc meetings are possible as well

each paper usually publishes its code, but check how old it is
ideally don’t write code from scratch, find what other people have done and build on that (less debugging, more building)
check hugging face (have pre-trained models and datasets) 

q: any papers on data efficiency? 
a: will send one (loosely related) on mattermost

read vtab paper

https://arxiv.org/pdf/2104.05704 uses a compact dataset

don’t obsess over hyperparameters, a reasonable set is good enough (if a paper uses batch size 64, try 32/128, not more)
early stopping is good (otherwise might overtrain) when the learning curve evens out (use more in fine-tuning than in pre-training, pre-training might still benefit from this)

! it’s important that the pre-training dataset has related knowledge to the fine-tuning

q: is it necessary for all results to be comparable? that would mean everyone has to do either pre-training or fine-tuning
a: you have a common goal, in the end each method is compared to all others, so it’s much better if it’s comparable, but not really necessary