# Multi-label Classification with Dense and Sparse Embeddings

This repository contains two notebooks that demonstrate the use of **dense embeddings** (using an embedding bag) and **sparse embeddings** (via TF-IDF) for multi-label classification tasks. Both approaches are applied using a **multi-layer perceptron (MLP)** neural network to classify text data into multiple categories. The results show clear differences in performance between the two embedding strategies, and the comparison provides useful insights into their strengths and limitations.

## Dataset

The dataset used in this project comes from a Kaggle competition where the task was to classify questions from **StackExchange** users into different tech domains. Since each post could mention multiple tech domains, this is a **multi-label classification** task. The domains cover a range of technologies, and each question may be tagged with more than one relevant technology.

### Dataset highlights:
- **Source**: Kaggle competition data.
- **Task**: Classifying questions into multiple tech domains.
- **Multi-label nature**: Each question can belong to multiple categories, making this an ideal dataset for exploring multi-label classification.
  
The goal of this project is to build a model using both **dense embeddings** and **sparse embeddings (TF-IDF)** to classify these questions. Along the way, I aim to analyze and compare the performance and benefits of these two embedding strategies in this context.

## Notebook: `dense_embeddings_stack_posts_nn`

In this notebook, I use **dense embeddings** with an **embedding bag** layer, which allows the model to learn and adapt word representations as it trains. This method helps the model better capture the relationships between words, resulting in improved classification performance.


### Key highlights:
- **Embedding Bag Layer**: The model leverages an embedding bag that learns the representations as it goes, making each word’s context more meaningful to the task at hand.
- **MLP with Layers**: A multi-layer perceptron (MLP) with layers of fun – hidden layers, ReLU activations, batch normalization (because we love stability), and dropout to keep overfitting at bay.
- **Performance**: With dense embeddings, the model evolves and learns as the training progresses. You’ll see improvements in **train and validation loss** as the model gets better at figuring out the semantic relationships in the data.
- **Generalization**: What makes this approach the best? The model can generalize to new data much better, thanks to the dynamic nature of dense embeddings.


### Results:
- **Validation loss and metric curves**:
  
  ![image](https://github.com/user-attachments/assets/ef56a99b-0c35-47b8-ba1d-d7a268425124)
  
  ![image](https://github.com/user-attachments/assets/9c12b272-ac93-4021-b1bd-6a9770a985ba)



- **Confusion matrix and Hamming Distance Scores**:

  ![image](https://github.com/user-attachments/assets/b3a4b4ae-5230-4910-91ea-d7479e0b1814)

  ![image](https://github.com/user-attachments/assets/261e704b-5cae-4d93-be4c-322674946882)

  Note: I chose these colors and the font within the notebook in the spirit of Halloween :)


## Notebook: `sparse_embeddings_stack_posts_nn`

In contrast to the dense embeddings, this notebook uses **TF-IDF sparse embeddings**. These are pre-calculated vectors based on word frequency, and they remain static throughout the training process. While TF-IDF can be useful for simpler models and tasks, it struggles to capture the semantic meaning behind words, which limits its effectiveness in this classification task.

### Key highlights:
- **TF-IDF embeddings**: These are **precomputed** and do not update during training, making them less adaptable to the data.
- **Simplified model setup**: Since the embeddings are static, we don’t need a collate function, and the model is slightly simpler.
- **Performance**: The model quickly plateaus in performance since it’s relying on frequency-based embeddings that don't learn or adapt during training.
- **Limitations**: While TF-IDF embeddings are useful for certain tasks, they are not ideal for tasks that require understanding **context** and **semantic relationships** in the text.

### Results:
- **Validation loss and metric curves**:

  ![image](https://github.com/user-attachments/assets/8bf2e109-dda4-46c9-a289-4fc08df17085)

  ![image](https://github.com/user-attachments/assets/1543b0b8-d412-4fd7-881c-36a9118bb327)


- **Confusion matrix**:

  ![image](https://github.com/user-attachments/assets/ddd3629f-0841-4ee0-90a1-9aa416bb464d)

  ![image](https://github.com/user-attachments/assets/51ab8fff-9779-4fc1-8e94-87662fe588ee)

  Note: I don't know why I chose these colors but it worked with the rest of the notebook.



## Lessons Learned

Through this comparison, it's clear that **dense embeddings** significantly outperform **sparse embeddings** like TF-IDF when it comes to capturing the semantic meaning of text. Dense embeddings provide a more nuanced understanding of the relationships between words, which leads to improved performance in classification tasks.

That said, **TF-IDF** still has its place for simpler tasks where word frequency is enough, but for multi-label classification and other complex NLP tasks, **dense embeddings** offer a much better solution.

## Next Steps
- Explore different types of **dense embeddings** (e.g., BERT, GloVe) to further improve the model's ability to classify text.
- Experiment with transfer learning by using pre-trained models and fine-tuning them on this specific dataset.
- Continue to optimize the model architecture to strike a balance between **accuracy** and **computational efficiency**.

Feel free to explore my notebooks for a detailed walkthrough of the model architectures, training processes, and results. 

- Happy Learning :)


