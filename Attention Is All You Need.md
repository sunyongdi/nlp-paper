# Attention Is All You Need

较好的序列转导模型都是基于包含编码器（encoder）、解码器（decoder）的复杂的循环神经网络（recurrent neural networks，RNN）、卷积神经网络（convolutional neural networks，CNN）。表现比较好的模型，会通过注意力机制（attention mechanism）连接编码器和解码器。
本文提出一种比较简单的网络结构（Transformer），Transformer完全基于注意力机制。
有实验表明：基于这个网络结构的模型，质量更高、更易并行化、训练时间更短。
本文进行了两个实验：WMT 2014 English-to-German、English-to-French机器翻译
评价指标：BLEU

![image-20231120143402785](https://picgo-1305561115.cos.ap-beijing.myqcloud.com/img/image-20231120143402785.png)

## **1. Introduction**

语言建模（language modeling）、机器翻译（machine translation）等序列建模（sequence modeling）和转导问题（transduction problem）也叫序列到序列（sequence to sequence）问题。目前最先进（sota）的方法是：RNN、长短期记忆（long short-term memory，LSTM）、门循环神经网络（gated recurrent neural networks）。
RNNs的问题：对输入和输出序列的符号位置进行计算，将位置和计算时间中的步骤对齐，生成一个隐状态$h_t$序列，这种固有的序列本质阻碍了**训练示例内的并行化**，但是这一点在较长序列中是很重要的，因为内存约束限制了示例之间的批处理。
最近的工作通过因子技巧（factorization tricks）和条件计算（conditional computation）在计算效率上有显著提升，条件计算还提升了模型表现。
但是序列计算的基本限制仍然存在。

Attention允许对dependencies进行建模，而无需考虑它们在输入和输出序列中的距离。但是，大多数情况下，Attention都与RNN结合使用。

Transformer，避免RNN，完全依赖Attention来刻画输入和输出之间的global dependencies。此外，Transformer允许更多并行化，并且达到了一个新的最先进的翻译质量。

## **2. 背景**

减少序列计算这个目标也构成了扩展神经GPU、ByteNet、ConvS2S的基础，它们都使用CNN作为基础构件，并行化计算所有输入输出位置的隐层表示。在这些模型中，关联来自两个任意输入或输出位置的信号所需的操作数随着位置之间的距离而增长，ConvS2S是线性增长、ByteNet是对数增长。因此，要学习远距离位置之间的dependencies就更困难了。在Transformer中，这个操作数被削减为了常数。尽管由于平均注意力-加权（attention-weighted）位置导致有效解析度削减，但我们可以使用3.2中的多头注意力（multi-head attention）抵消这个影响。

自我注意力（self-attention），也称内在注意力（intra-attention），通过**关联单个序列的不同位置**来计算序列表示。这种注意力机制已经成功应用于阅读理解（reading comprehension）、摘要总结（abstractive summarization）、文本蕴含（textual entailment）、学习独立于任务的句子表示。

端到端记忆网络（end-to-end memory networks）是基于**循环注意力机制**而不是序列对齐循环，并且实验证明在简单语言问答和语言建模任务上表现很好。

Transformer是第一个完全靠self-attention来计算输入和输出representation的转导模型，而不使用序列对齐的RNN或CNN。

## **3. 模型结构**

大多数有竞争力的神经序列转导模型都有编码器-解码器结构。在此，编码器把输入序列（符号表示($x_1,...x_n$)）映射为连续表示序列$z=(z_1,...z_n)$。给定$z$，解码器就会生成输出序列$(y_1,...y_m)$（一次生成一个符号）。模型的每一步都是自回归，即当生成下一个符号时会使用之前生成的符号作为附加输入。

Transformer遵循这种总体架构，对编码器和解码器都使用stacked self-attention、point-wise、fully connected layers。

<img src="https://picgo-1305561115.cos.ap-beijing.myqcloud.com/img/image-20231121182634997.png" alt="image-20231121182634997" style="zoom:50%;" />

### **3.1 编码器和解码器堆（encoder and decoder stacks）**

Encoder：编码器由N=6个相同的layers堆叠而成，每一层都有2个子层（sub-layer）。第一个子层是multi-head attention，第二个子层是一个简单的、position-wised 全连接前馈神经网络（FFNN）。我们对每一个子层都采用残差连接（residual connection），然后进行层归一化（layer normalization）。也就是说，每一个子层的输出都是���������(�+��������(�))，其中��������(�)是子层自身实现的函数。为了方便，模型中的所有子层以及嵌入层，都控制输出维度������=512。

