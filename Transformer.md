## Basic Concept
### Transformer Architecture
![transformer archetecture](./imgs/Annotated-Transformers-Architecture.webp)

The encoder maps an input sequence of symbol representations (x1,...,xn) to a sequence of continuous representations z = (z1,...,zn). Given z, the decoder then generates an output sequence (y1,...,ym) of symbols one element at a time. At each step the model is auto-regressive consuming the previously generated symbols as additional input when generating the next.

**Encoder**: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512. 

**Decoder**: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

### Self-attention

 Self-attention computes a weighted sum of all input elements, where the weights are determined dynamically based on their inter-relationships. This allows capturing dependencies between distant words in a sentence.

   – In order to determine the weights dynamically, the self-attention mechanism computes three vectors for each input element: Key (k), Query (q), and Value (v). These vectors are then used to compute the weights of different words in the input sequence

   ![self attention calculation](./imgs/transformers-self-attention-step-by-step-explanation.webp)


The role of **Multi-Head Attention** is to capture different types of relationships in the data. This is accomplished by using multiple self-attention heads in parallel, whereby each head learns different aspects of the input, allowing the model to attend to various patterns simultaneously.

## GPT (Generative Pretrained Transformer)

[GPT Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) demonstrate that large gains on these tasks can be realized by generative pre-training of a language model onadiverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task. In contrast to previous approaches, we make use of task-aware input transformations during fine-tuning to achieve effective transfer while requiring minimal changes to the model architecture.

We employ a **two-stage training procedure**. First, we use a language modeling objective on the unlabeled data to learn the initial parameters of a neural network model. Subsequently, we adapt these parameters to a target task using the corresponding supervised objective.

**Why only Decoder??**
in case of translating from one language to another, we know the full context from the original sentence, so basically before running the decoder to predict the next part, we first encode the context from the original language. Or say encoding is more about encoding what the data means into symbolic representation, while decoding is more about predicting the output - the original transformer did both of these things, while gpt only does the second part

![GPT training](./imgs/GPT_training.png)


[GPT Paper2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) -> demonstrates language models can perform down-stream tasks in a zero-shot setting– without any parameter or architecture modification. We demonstrate this approach shows potential by highlighting the ability of language models to perform a wide range of tasks in a zero-shot setting.

## BERT vs GPT
![bert vs gpt](./imgs/compare.png)

