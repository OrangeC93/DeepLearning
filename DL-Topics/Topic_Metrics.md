# Triplet Loss 
1. Siamese Networks: Siamese networks are neural networks containing two or more **identical** subnetwork components.
	- Main idea: Siamese Network can learn useful data descriptors that can be further used to compare between the inputs of the respective subnetworks. It is important that not only the architecture of the subnetworks is identical, but the weights have to be shared among them as well for the network to be called “siamese”.
	- Inputs can be anything from numerical data, image data or even sequential data such as sentences or time signals.
	- How to train: you can train the network by taking an anchor image and comparing it with both a positive sample and a negative sample. The dissimilarity between the anchor image and positive image must low and the dissimilarity between the anchor image and the negative image must be high.

2. Triplet Loss: l = max(d(a,p)- d(a,n)+m, 0)
	```
	Hereby, d is a distance function (e.g. the L2 loss), a is a sample of the dataset, p is a random positive sample and n is a negative sample. m is an arbitrary margin and is used to further the separation between the positive and negative scores. i.e if margin = 0.2 and d(a,p) = 0.5 then d(a,n) should at least be equal to 0.7. Margin helps us distinguish the two images better.
	```

Reference: (https://becominghuman.ai/siamese-networks-algorithm-applications-and-pytorch-implementation-4ffa3304c18, https://towardsdatascience.com/siamese-network-triplet-loss-b4ca82c1aec8)
# BLEU Score
1. BLEU compares the machine-written translation to one or several human-written translation and computes a similirity score based on:
	- n-gram precision (usually for 1,2,3,4-grams)
	- plus a penalty for too-short system translation
2. BLEU is useful but imperfect
	- many valid way to translate a sentense
	- so a good translation can get a poor BLEU score because it has low n-gram overlap with the human translation
# Maxout Networks([Explanation](https://watsonyanghx.github.io/2017/03/10/Maxout-Networks-Network-in-Network/))
1. Architecture: 每个隐藏层的hidden units与前一层的hidden units连接的时候，加入一个有k（是个超参数）个hidden units的隐藏层，一般叫做“隐隐藏层”
	
	![model overview](/pictures/maxout.jpg)
2. Advantages
	- Maximum operation ensure non-linear property (因为是 max 操作，可以保证非线性)
	- Approximate any convex function (同时可以拟合任何的凸函数，论文中用数学的方法给出了证明，不过它只能拟合凸函数，算是一个弊端，后面的Network in Network对这里做了改进)
	- Filter can be trained to learn (每个hidden units可以学习各自的activation function形式)
3. Difference: ReLU使用的 max(x,0) 是对每个通道的特征图的每一个单元执行的与0比较最大化操作；而maxout是对每个通道的特征图在通道的维度上执行最大化操作（Cross chanel pooling).
4. Disadvantage: Parameters increased by k times 主要原因是因为参数会成倍的增加。不过还是会见到有用的，比如GAN的 discriminator 就用了.
5. Movivation: dropout原理， 如果不只是单单的把它作为一种提升模型性能的工具，而是直接应用到网络结构的内部，是不是能够更好的提高网络的性能呢？
# Support Vector Machines
	+ Maximal-Margin Classifier
	+ Kernel Trick
# PCA ([Explanation](https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues/140579#140579))
	+ PCA using neural network
		+ Architecture
		+ Loss Function
# Spatial Transformer Networks
1. Theory: STN helps to crop out and scale-normalizes the appropriate region, which can simplify the subsequent classification task and lead to better classification performance.
	![model](/pictures/snt.png)
# Gaussian Mixture Models (GMMs)
# Expectation Maximization
# Perplexity

In English, the word 'perplexed' means 'puzzled' or 'confused' (source). Perplexity means inability to deal with or understand something complicated or unaccountable.

- Perplexity is a measurement of how well a probability model predicts a test data. In the context of Natural Language Processing, perplexity is one way to evaluate language models.
- Perplexity is just an exponentiation of the entropy!
- Low perplexity is good and high perplexity is bad since the perplexity is the exponentiation of the entropy