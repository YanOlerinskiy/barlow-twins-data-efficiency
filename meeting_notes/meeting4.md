#CSE3 #research 

### Status Updates

- Max
	- Need to add random initialization accuracy after linear probing (baseline to show the improvement pre-training gives)
	- No blockers
- Yan
	- Main blocker: GPU renting overhead
	- Main adapter code is done
- Dimo
	- Splits up to 64k done
	- Experiment: Tried fixing the number of gradient steps between the splits
		- Little difference on cifar-10 (within 1%)
		- ImageNet inline linear probing accuracies are **quite** different
	- No blockers
- Leo
	- Splits up to 32k done
	- 1k performance consistent with Max
		- Tested random initialization accuracy - $~36\%$
		- 1k performed marginally worse
	- Main blocker - training takes a WHILE
- Makar
	- MoCo adaptation work done
	- Linear probe accuracy
		- Randomly initialized transformer giving 45% (what how?)
		- 2k accuracy ~52.5%, 4k accuracy ~56.5%
	- Main blocker - pretraining times
	- Applied to dyke (cluster) - haven't replied yet
		- They might be on vacation
	- Question: should we decrease the number of parameters in the ViT?
		- Might be a research question for after

#### Discussion Round

##### Why could our (hypothetical, combined) paper could be rejected to a conference
- Could be rejected because of implementation discrepancies (support for that - random initialization linear probe accuracy doesn't align between Leo and Makar)
- Lack of clear conclusions
	- We could go for something similar as MNIST-1D
	- Refresher: they showed the properties of architectures were preserved on lower scale, allowing to do a lot of experiments in a smaller time frame

##### How would a small company do hyperparameter tuning on one of our models?
- Tune the learning rate and epochs
	- Tune them per-model, even per-split (no need to keep it fair)


### Our Questions

- Epoch / iterations heuristic
	- We have options
	- Jan: could investigate how SSL loss relates to downstream tasks (opposed to downstream validation quality)
		- MAE paper could have that
- Should we share the code for training?
	- Potentially, but after we get the initial curves, as it won't influence our grade
- Moving to data splits of powers of 2
	- Jan: small companies wouldn't want to throw out the data they have access to
	- Jan: also, this only really affects the smaller splits
		- So it wouldn't really make the result worse if we make batch size 500 there, so that all the data is used, essentially
	- Conclusion: it's probably fine to set batch size not to power of 2, as long as we don't throw out too much pre-training data
- Makar's random initialization accuracy after linear probing is ~45%
	- Need to compare the linear probing stack, or cifar-10 loader for each of us
	- Likely not some super-lucky shuffling seed