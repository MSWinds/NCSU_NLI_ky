# Modified NLI BART Model with sliding windows

## Description
---

This project addresses the challenge of assessing the factuality of system-generated feedback. Our primary objective is to design a "filter" capable of autonomously determining the factuality of generated statements, ensuring that students receive only pertinent and factual feedback. 

A widely recognized strategy for this task is the utilization of Natural Language Inference (NLI) models. Specifically, the BART model stands out as a significant representative in this domain. NLI tasks focus on discerning whether a given "hypothesis" is <u>true (entailment)</u>, <u>false (contradiction)</u>, or remains undetermined (neutral) based on a "premise."

In this enhanced version of the notebook, tailored for Google Colab, users will experience several key improvements:
- Implementation of a **sliding window** technique for handling longer sequences.
- Modifications to the foundational **BART model structure** to optimize performance.
- Removal of **redundant parts** related to label calculations inside the model, streamlining the process.
- Noticeably **enhanced performance** in comparison to the previous version. However, it's important to note that this comes at the expense of increased computational costs and extended processing times.

Embark on a journey that encompasses data preprocessing, model configuration, training, and evaluation to implement an NLI model for assessing factuality.

## Usage
---

### **Data Preparation**

We are using the dataset called `Gen_review_factuality.csv`. The dataset should have the following format:


Make sure your dataset adheres to this format before proceeding.
```python
pid object
doc object
feedback object
label int64
```
### **Parameter Configuration**

In the notebook, there is a "Parameter Configuration" cell where you can adjust various hyperparameters and parameters for the model. The parameters include:

```python
bs = 1 # set the batch size
learning_rate = 1e-6 # set the initial learning rate 
weight_decay = 1e-3 # set the initial weight_decay
num_epochs = 5 # set the number of epochs
weight_ratio = 2.718 # weight adjustment based on the inverse ratio of 0 and 1
```
Notice that the number of chunks for the **sliding windows** can be adjusted in the " ModifiedBartForSequenceClassification" class. The default value is 4.


### **Modifications to the BART Model**
---
In the enhanced version of the notebook, the foundational BART model has undergone significant modifications to cater to the specific needs of our application. Below are the highlights of the modifications introduced:

This class offers a custom implementation of the BART model tailored for sequence classification tasks. The key modifications include:

1. **Embedding Fusion Layer**:
   - An additional linear layer, `embedding_fusion_layer`, has been introduced. This layer fuses the embeddings from multiple chunks of input data. The model can handle input sequences split into multiple chunks (currently set to 4). The embeddings from each chunk are concatenated and then passed through this fusion layer.
   
2. **Chunk-wise Processing**:
   - The `forward` method of the model has been modified to process input data in chunks. For each chunk, the BART model generates embeddings, which are later concatenated.
   
3. **Sentence Representation Extraction**:
   - After fusing the embeddings from all chunks, the model extracts the sentence representation using the `eos_token` as in the original BART model. If no `eos_token` is found in a chunk, the last token's embedding of that chunk is used.
   
4. **Logits Computation**:
   - The final sentence representation is passed through a classification head to compute logits for the sequence classification task.

5. **Remove Loss Calculation from the Model**:
   - The `ModifiedBartForSequenceClassification` model was designed to calculate the loss if labels are provided. However, in the enhanced approach, the loss calculation has been shifted to the training loop. This modification not only speeds up the model's forward pass but also simplifies the model's output structure.

6. **Simplify Model Outputs**:
   - With the removal of the loss calculation from the model, the outputs have been streamlined. The model now returns only the logits or a dictionary containing the logits, making it easier to handle the outputs in the training loop.

Here is a high-level overview of the class:

```python
class ModifiedBartForSequenceClassification(BartPretrainedModel):
    ...
    def __init__(self, config):
        ...
        self.embedding_fusion_layer = nn.Linear(config.d_model * 4, config.d_model)
        ...

    def forward(
        self,
        ...
    ):
        ...
        for i in range(num_chunks):
            ...
        concatenated_embeddings = torch.cat(chunk_embeddings, dim=-1)
        ...
        logits = self.classification_head(sentence_representation)
        ...
```

### **Introducing Sliding Window Approach**:
---
To handle longer sequences that exceed the model's maximum token limit, a sliding window approach is implemented. The key components of this approach include:

**1. Chunk Creation**:
   
- `create_chunks` function combines the document (`doc`) and feedback (`hypothesis`) and breaks them into overlapping chunks of tokens.
- Each chunk ensures that it captures a portion of the text and hypothesis, with overlaps defined to ensure continuity.
- The function returns a list of chunks, with each chunk having a defined size (`chunk_size`). The overlap between chunks is controlled by the `overlap` parameter.
- If the total chunks created are less than the `max_chunks`, the function pads the chunk list. If more, it truncates the list.

**2. FeedbackDataset Class**:
   
- The Dataset class (`FeedbackDataset`) has been adapted to use the chunking mechanism.
- The `__getitem__` method retrieves a data point, creates chunks for it using the above-defined function, and returns the chunks as input ids.
- If a chunk is empty (which might be due to some specific input causing the tokenizer to behave unexpectedly), the method prints out the problematic document and feedback to aid debugging.

**3. Batch Collation**:

- A custom batch collation function (`collate_batch`) has been defined.
- This function processes the chunks created for each data point in the batch, concatenates them, and reshapes them into the desired shape for training.
- It returns the input ids, attention masks, and labels for the entire batch.


### **Modification to the Training and Testing Loop**
---
To optimize the training and testing loops and ensure a seamless flow of data through the `ModifiedBartForSequenceClassification` model, the following changes have been implemented:

   **Label Extraction and Removal**:
   - The method `batch.pop("labels")` is employed to achieve a dual purpose: extracting the labels from the batch and simultaneously removing them. This ensures that when passing `**batch` to `nli_model`, the batch doesn't contain the labels. However, the extracted labels tensor is retained, making it available for subsequent loss computation.

This modification provides a streamlined and efficient way of handling labels during the training and testing phases, ensuring that only necessary inputs are passed to the model while retaining all essential data for performance evaluation.

### **Troubleshooting**
---
If you encounter the `OutOfMemoryError: CUDA out of memory` error message, consider the following troubleshooting steps:

1. **Refresh GPU Connection**:
   - Try refreshing and reconnecting to the GPU. This can sometimes clear up any lingering memory issues.
   
2. **Reduce Batch Size**:
   - Set the batch size to 1. This can significantly reduce memory requirements at the cost of potentially slower training.

3. **Modify Model Output**:
   - Remove the computation of the neutral probability in the model. This can save some memory as it reduces the size of the model's output.

4. **Model Saving**:
   - Consider alternative methods for saving the best model state. Instead of using deep copy, which can be memory-intensive, look for lightweight solutions.

5. **Upgrade GPU**:
   - If the above steps don't resolve the issue, consider switching to a GPU with more memory. This can be particularly helpful for large models or datasets.


### **Limitations**
---
While the enhanced version of the notebook has introduced several improvements, it's essential to acknowledge certain limitations:

1. **Increased Memory Usage and Computation Time**:
   - The introduction of the sliding windows technique, though effective for handling longer sequences, has led to a surge in memory usage and computation time. This could potentially pose challenges when working with limited computational resources.

2. **Number of Chunks for Sliding Windows**:
   - The current implementation utilizes a fixed number of chunks for the sliding windows. However, there's room for experimentation with different values to potentially optimize performance further. Finding the optimal number of chunks is crucial for balancing computational efficiency and model performance.

3. **Handling Long Text**:
   - While the sliding window technique offers a viable solution to the long text challenge, there are several other methods in the NLP domain that could be explored. Incorporating alternative strategies might offer different perspectives and potentially better results when dealing with lengthy texts.
   - 
4. **Hyperparameter Fine-Tuning**:
   - With the modifications and enhancements introduced in this version, further hyperparameter fine-tuning might be necessary to extract the maximum potential from the model. The improved model's dynamics might require different hyperparameter values for optimal performance.

