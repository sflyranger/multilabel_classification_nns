# Multi-label Classification with Dense and Sparse Embeddings

This repository contains two notebooks that explore the use of **dense embeddings** and **sparse embeddings (TF-IDF)** in a multi-label classification task. Both approaches are applied using a multi-layer perceptron (MLP) neural network, and the results are compared to see how each type of embedding impacts performance.

## Notebook: `dense_embeddings_stack_posts_nn`

In this notebook, I utilize **dense embeddings** for the multi-label classification task. Dense embeddings, like Word2Vec or pre-trained embeddings such as BERT, are learned representations that capture **semantic relationships** between words. This allows the model to gain a deeper understanding of the context within the text, which typically leads to better performance compared to static, frequency-based embeddings like TF-IDF.

### Key highlights:
- **Embedding layer**: The model learns or uses pre-trained dense embeddings, allowing for richer representations of the input text.
- **Neural network structure**: A multi-layer perceptron (MLP) with multiple hidden layers, activation functions, batch normalization, and dropout to prevent overfitting.
- **Model performance**: The model shows significant improvements in both **train** and **validation loss** as it learns from the dense embeddings.
- **Generalization**: Dense embeddings enable better generalization when applied to unseen data, making this a preferred approach for semantic tasks.

### Results:
- **Validation loss and metric curves**:
  
  ![image](https://github.com/user-attachments/assets/ef56a99b-0c35-47b8-ba1d-d7a268425124)
  ![image](https://github.com/user-attachments/assets/9c12b272-ac93-4021-b1bd-6a9770a985ba)



- **Confusion matrix**:
  
 

## Notebook: `sparse_embeddings_stack_posts_nn`

In contrast to the dense embeddings, this notebook uses **TF-IDF sparse embeddings**. These are pre-calculated vectors based on word frequency, and they remain static throughout the training process. While TF-IDF can be useful for simpler models and tasks, it struggles to capture the semantic meaning behind words, which limits its effectiveness in this classification task.

### Key highlights:
- **TF-IDF embeddings**: These are **precomputed** and do not update during training, making them less adaptable to the data.
- **Simplified model setup**: Since the embeddings are static, we don’t need a collate function, and the model is slightly simpler.
- **Performance**: The model quickly plateaus in performance since it’s relying on frequency-based embeddings that don't learn or adapt during training.
- **Limitations**: While TF-IDF embeddings are useful for certain tasks, they are not ideal for tasks that require understanding **context** and **semantic relationships** in the text.

### Results:
- **Validation loss and metric curves**:

  *(Add validation loss and metric curves here for `sparse_embeddings_stack_posts_nn`)*

- **Confusion matrix**:

  *(Add confusion matrix images here for `sparse_embeddings_stack_posts_nn`)*

## Lessons Learned

Through this comparison, it's clear that **dense embeddings** significantly outperform **sparse embeddings** like TF-IDF when it comes to capturing the semantic meaning of text. Dense embeddings provide a more nuanced understanding of the relationships between words, which leads to improved performance in classification tasks.

That said, **TF-IDF** still has its place for simpler tasks where word frequency is enough, but for multi-label classification and other complex NLP tasks, **dense embeddings** offer a much better solution.

## Next Steps
- Explore different types of **dense embeddings** (e.g., BERT, GloVe) to further improve the model's ability to classify text.
- Experiment with transfer learning by using pre-trained models and fine-tuning them on this specific dataset.
- Continue to optimize the model architecture to strike a balance between **accuracy** and **computational efficiency**.

Feel free to explore the notebooks for a detailed walkthrough of the model architectures, training processes, and results.


