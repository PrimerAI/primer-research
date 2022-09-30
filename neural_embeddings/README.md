# Neural Embeddings

# Neural Embeddings

Neural embeddings are introduced in [Neural Embeddings for Text](https://arxiv.org/abs/2208.08386)

Usage is simple: create `EmbedNeural`, and use `make_embedding`, which takes single argument - text - and returns embedding. The embedding is returned as a torch vector, because the downstream operatins may be in torch. The embedding dimension is sum of sizes of the selected layers, the default is 768 * 3 = 2304. When creating `EmbedNeural`, set up n_epochs to 1 or higher, keeping in mind the tradeoff between the execution time and quality. Similarly, the option 'keep_hidden_states' allows faster processing, with minor difference in quality. (Also consider which batch_size works for you.) `EmbedNeuralTop` is a simplified faster version. 
