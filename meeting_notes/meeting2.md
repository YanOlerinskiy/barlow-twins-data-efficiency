#CSE3 #research 

### Shared Data Pipeline

>Loads Tiny ImageNet, splits into 1, 2, 4, ..., 100k images. 200 images.

Mentor Input:
- Some slices might be too small ($\implies$ won't lead to valuable models after training)
- Sharing the GitHub repo link with the mentors is fine

### Methods' Division

- Yan - Barlow Twins
	- Original paper is open to more augmentations - potential vector for improvements
	- Concern: A bit similar to DiNO, is that fine?
		- Mentor Input: a little overlap is not a problem, and the optimizations will likely be different
- Leo - DiNO
	- Question: so many hyperparameters, how (what) do we tune?
		- Embedding size
			- Keep embedding size small for local computation
		- Model architecture for the backbone - how do we choose?
			- Mentor Input: Just choose one and stick with it, tuning is hard (ResNet or ViT)
			- Perfect setup isn't the goal anyway
			- For the project to be feasible, it should stay within the day of training
			- The final goal is data efficiency, so don't tune for faster training
- Makar - MoCo
	- Similar to SimCLR, but optimized, allows increasing batch size
	- *Not really any questions there*
- Max - I-JEPA
	- Mentor Input: Try running it on the validation set from Tiny Imagenet
		- Make sure to keep the 80-20 ratio so there's no leak
- Dimo - MAE
	- Tried implementing a training script
	- *Not really any questions there*

### Technical Questions

Models have "underlying methods" as a parameter often
- Transformers are said to require much bigger training set sizes
- Transformers have "context windows", and for these context windows to make sense, we might need to make it bigger
	- We're currently using 64x64 images, with the context of 8
	- Mentor Input: Tiny Transformer is probably fine
		- Would be nice seeing this difference for transformers and CNNs
	- Yan: start with transformer, switch to CNN if it sucks
	- Mentors: Preferable to use the same architecture

Compute
- Makar got access to the cluster, we get low priority - but that's still access
- Can also apply to [DAIC](https://daic.tudelft.nl)

Finetuning
- Decision btw. PEFT or Linear Probing
	- Mentors: PEFT would usually work better, so let's use it as starting point ig

### Presentation Slides

>All points below are mentors' feedback

- Don't have to show the presentation(s) to the professor, he only really cares about the findings
- Research question - meh?
	- Group question is rather "how do the methods compare?", not "how do we evaluate"
	- How should we phrase the individual research question?
		- Start with something like: "How can we optimize *method* for data efficiency?". When it becomes clearer during the research, make it grainier.
			- So like "which augmentations allow me to not collapse?", etc.
- Some later slide has: "Sanity checks against public checkpoints" - what's that?
	- Comparing learning patterns with a model with different parameters, trained on a differnet dataset - that's probably irrelevant
	- Checking btw. the peers is probably more relevant here
- PEFT evaluation - why only the subset of VTAB?
	- Makar: generalistic dataset will probably have 0 overlap with medical data / satellite imagery. Does it make sense to use them?
		- It's fine if the generalistic dataset doesn't overlap with VTAB benchmark
		- This is the goal - learn general patterns from pre-training, learn domain-specific stuff from finetuning
	- Maks: what about some VTAB datasets being 6x bigger than Tiny Imagenet?
		- Look into VTAB-1k not to use massive datasets for the finetuning
		- Different resolutions are also fine, just rescale to 64x64 (probably already done in the implementation by default)
- The research plan is reasonable
	- Two weeks for the writing is doable
	- Enough time for experiments as well
- Next week
	- Tuesday 1:00 (EDIT: building closed, rescheduled to Wednesday)

