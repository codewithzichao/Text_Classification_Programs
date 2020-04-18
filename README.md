## 前言

文本分类是NLP中最为基础的任务，同时也是其他生成式任务的基础。本repo旨在使用tensorflow2实践常见文本分类模型(textCNN，RCNN，HAN等等)，加深对文本分类的理解，也是对之后从事dialogue打下基础。let‘s start！🤩

### textCNN

**Reference：**[Convolutional Neural Networks for Sentence Classiﬁcation](https://arxiv.org/abs/1408.5882)

textCNN模型架构其实非常好懂。如下图所示。👇
![](/Users/codewithzichao/MyGithubProjects/Project5/Text_Classification_Programs/images/textCNNjpg.jpg)

输入的形状是:**(batch_size,seq_length,embedding_size)**；然后使用多个不同大小的卷积核(2，3，5)，然后进行一维卷积操作，具体来说，卷积核的形状是：**(2,embedidng_size,channels)**，chaanels就是卷积核的数量；进行卷积操作之后，我们得到的形状大小会不一样，所以，我们需要使用GlobalMaxPooling操作，只保存得到的向量中的最大值，之后进行拼接，最后接上一个softmax层，得到最后的分类结果。

![]()

