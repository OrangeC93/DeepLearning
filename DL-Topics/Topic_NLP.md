# Natural Language Processing (My mission is to expand each point listed here)

# Recurrent Neural Networks
1. Architectures (Limitations and inspiration behind every model) ([Blog 1](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)) ([Blog 2](https://colah.github.io/posts/2015-08-Understanding-LSTMs/))
2. Vanilla RNN: 
	- [Disadvantages][https://medium.com/learn-love-ai/the-curious-case-of-the-vanishing-exploding-gradient-bf58ec6822eb): 
		- vanishing or exploding gradient problem
		- forget the earlier input
3. Identity RNN
3. GRU
4. LSTM: given A, .. _, use long time ago data to infer missing part.
	- Learn Gate: combines STM + E (input) and chooses to ignore the unneeded info.
	- Forget Gate: dump out all the unnecessary long term information. 
	- Remember Gate = New LTM: learn gate output + forget gate output
	- User Gate = New STM: takes the LTM from the forget gate, and STM + E from the learn gate and uses them to come up with a new short term memory or an output (same thing).
5. Bidirectional: given A, _ , C , use A and C to infer the missing part. 
6. ResNet
6. Vanishing and Exploding Gradients
# [Word Embeddings](http://hunterheidenreich.com/blog/intro-to-word-embeddings/) ([pratical cases](https://github.com/zlsdu/Word-Embedding))
	- 简介: 
		- 统计方法：通过统计词语之间的关系，定义一些显性隐性的关系，从而构建词向量。例如SVD，LSA等等。这样的做法在逻辑上不理性，效果上也不好。
		- 语言模型：通过构建语言模型来实现对词向量的学习，在理论上可行的，并且目前大部分的工作就是基于这样的思想。从最开始的神经网络语言模型（NNLM）到后来的Word2vec，GloVe等等。
1. One-Hot Encoding (Count Vectorizing): Really huge and sparse vectors that capture absolutely no relational information. 
2. [TF-IDF Transform](https://zhuanlan.zhihu.com/p/31197209)
   - ![model overview](/pictures/TFIDF.JPG)
3. Co-Occurrence Matrix: Super large representation! If we thought that one-hot encoding was high dimensional, then co-occurrence is high dimensional squared. That’s a lot of data to store in memory.
4. Word2Vec:
    - Word2vec is a 2-layer neural network structure to generate word embedding by training the model on a supervised classification problem. 
    - [Two training algorithm]((https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92). ): 
		- CBOW (continuous bag of words ): CBOW is using context to predict a target word.
		- SG (skip-gram): skip-gram is using a word to predict a target context. 
		- Both of these learning methods use local word usage context (with a defined window of neighboring words). The larger the window is, the more topical similarities that are learned by the embedding. Forcing a smaller window results in more semantic, syntactic, and functional similarities to be learned.
		- Generaly, the skip-gram method can have a better performance compared with CBOW method, for it can capture two semantics for a single word. For instance, it will have two vector representations for Apple, one for the company and another for the fruit.
    - How it works: Word2vec is a predictive model which learns their vectors in order to improve their predictive ability of Loss(target word | context words; Vectors), i.e. the loss of predicting the target words from the context words given the vector representations, which is cast as a feed-forward neural network and optimized as such using SGD, etc.
    - Advantages: Word2Vec shows that we can use a vector (a list of numbers) to properly represent words in a way that captures semantic or meaning-related relationships (e.g. the ability to tell if words are similar, or opposites, or that a pair of words like “Stockholm” and “Sweden” have the same relationship between them as “Cairo” and “Egypt” have between them) as well as syntactic, or grammar-based, relationships (e.g. the relationship between “had” and “has” is the same as that between “was” and “is”).
    - Disadvantage: Word2vec only takes local contexts into account and **does not take advantage of global context**. 

    ![model overview](/pictures/word2vec.png)
   
    [Example](https://towardsdatascience.com/word2vec-from-scratch-with-numpy-8786ddd49e72)
	[Example](https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92)
4. Glove: [Global Vectors for Word Representation](https://juejin.im/post/5d609fdd6fb9a06b1777bfb9)
	- GloVe is a count-based model which learns their vectors by doing dimensionality reduction on the co-occurrence information. Firsly, construct a large matrix of (word * context) co-occurrence information. Secondly, factorize this matrix to a lower-dimensional (word * features) matrix, where each row yield a vector representation for each word. In the specific case of GloVe, the counts matrix is preprocessed by normalizing the counts and log-smoothing them.
	- 原理: 它是一个基于全局词频统计（count-based & overall statistics）的词表征（word representation）工具，它可以把一个单词表达成一个由实数组成的向量，这些向量捕捉到了单词之间一些语义特性，比如相似性（similarity）、类比性（analogy）等。我们通过对向量的运算，比如欧几里得距离或者cosine相似度，可以计算出两个单词之间的语义相似性。
	- 如何实现:
		- 构建共现矩阵: 根据语料库（corpus）构建一个共现矩阵（Co-ocurrence Matrix）X，矩阵中的每一个元素 Xij 代表单词 i 和上下文单词 j 在**特定大小的上下文窗口（context window）内**c共同出现的次数。一般而言，这个次数的最小单位是1，但是GloVe不这么认为：它根据两个单词在上下文窗口的距离 d，提出了一个衰减函数（decreasing weighting: decay=1/d 用于计算权重，也就是说距离越远的两个单词所占总计数（total count）的权重越小。
		- 词向量和共现矩阵的近似关系: 构建词向量（Word Vector）和共现矩阵（Co-ocurrence Matrix）之间的近似关系，论文的作者提出以下的公式可以近似地表达两者之间的关系: w是要求解的向量，b是bias term, label就是以上公式中的 log(Xij), w是要不断更新学习的参数，训练方式跟监督学习的训练方法没什么不一样，都是基于梯度下降的。
			- 公式 loss = f(X) [WiWj + bi + bj - log(Xij)]
		- 构造损失函数: 需要加个权重函数
	- 对比:
		- LSA（Latent Semantic Analysis）是一种比较早的count-based的词向量表征工具，它也是基于co-occurance matrix的，只不过采用了基于奇异值分解（SVD）的矩阵分解技术对大矩阵进行降维，而我们知道SVD的复杂度是很高的，所以它的计算代价比较大。还有一点是它对所有单词的统计权重都是一致的。而这些缺点在GloVe中被一一克服了。
		- [word2vec](https://www.quora.com/How-is-GloVe-different-from-word2vec)最大的缺点则是没有充分利用所有的语料，所以GloVe其实是把两者的优点结合了起来。
	- ![embedding example](/pictures/GloVe_embedding_example.png) 
3. FastText
	- 原理:
   		- 字符级别的n-gram: 对于单词“apple”，假设n的取值为3，则它的trigram有 “<ap”, “app”, “ppl”, “ple”, “le>”. 有连个好处:
		    - 对于低频词生成的词向量效果会更好。因为它们的n-gram可以和其它词共享。
        	- 对于训练词库之外的单词，仍然可以构建它们的词向量。我们可以叠加它们的字符级n-gram向量。
   - 模型架构: 和CBOW一样，fastText模型也只有三层：输入层、隐含层、输出层(Hierarchical Softmax). 
		- 输入都是多个经向量表示的单词，输出都是一个特定的target，隐含层都是对多个词向量的叠加平均。
		- 不同的是:
			- CBOW的输入是目标单词的上下文，fastText的输入是多个单词及其n-gram特征，这些特征用来表示单个文档.
			- CBOW的输入单词被onehot编码过，fastText的输入特征是被embedding过.
			- CBOW的输出是目标词汇，fastText的输出是文档对应的类标。
		- 值得注意的是，fastText在输入时，将单词的字符级别的n-gram向量作为额外的特征；在输出时，fastText采用了分层Softmax，大大降低了模型训练时间。
   - 核心思想：将整篇文档的词及n-gram向量叠加平均得到文档向量，然后使用文档向量做softmax多分类。这中间涉及到两个技巧: 字符级n-gram特征的引入以及分层Softmax分类。
   - 分类效果: 
   	- 使用词embedding而非词本身作为特征，这是fastText效果好的一个原因；
	- 另一个原因就是字符级n-gram特征的引入对分类效果会有一些提升 。
    - 举例: **我 来到 达观数据** 和 **俺 去了 达而观信息科技**
		- 这两段文本意思几乎一模一样，如果要分类，肯定要分到同一个类中去。但在传统的分类器中，用来表征这两段文本的向量可能差距非常大。传统的文本分类中，你需要计算出每个词的权重，比如tfidf值， “我”和“俺” 算出的tfidf值相差可能会比较大，其它词类似，于是，VSM（向量空间模型）中用来表征这两段文本的文本向量差别可能比较大。但是fastText就不一样了，它是用单词的embedding叠加获得的文档向量，词向量的重要特点就是向量的距离可以用来衡量单词间的语义相似程度，于是，在fastText模型中，这两段文本的向量应该是非常相似的，于是，它们很大概率会被分到同一个类中。
4. LSA(Latent Semantic Analysis)
4. SkipGram, NGram
5. ELMO
7. BERT ([Blog](http://jalammar.github.io/illustrated-bert/))
8. VAE(Variational Autoencoders) ([Material](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf))
	- What's VAE: VAEs are powerful generative models, having applications as diverse as from generating fake human faces, to producing purely synthetic music.
	- Why VAE: When using generative models, you could simply want to generate a random, new output, that looks similar to the training data, and you can certainly do that too with VAEs. But more often, you’d like to alter, or explore variations on data you already have, and not just in a random way either, but in a desired, specific direction. This is where VAEs work better than any other method currently available.

# Transformers ([Paper](https://arxiv.org/abs/1706.03762)) ([Code](https://nlp.seas.harvard.edu/2018/04/03/attention.html)) ([Blog](http://jalammar.github.io/illustrated-transformer/))
9. Universal Sentence Encoder
