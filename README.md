# NLI BART Model

## Description
---
This project aims to address the challenge of assessing the factuality of system-generated feedback. The primary goal is to implement a "filter" that can automatically determine the factuality of generated statements, ensuring that only pertinent and factual feedback is delivered to students. 

A prominent approach to achieve this is through the use of Natural Language Inference (NLI) models, with BART being a significant representative. NLI tasks involve evaluating if a given "hypothesis" is <u>true (entailment)</u>, <u>false (contradiction)</u>, or undetermined (neutral) based on a "premise." 

Within the confines of this notebook, users will harness the power of the BART model tailored for Google Colab, guiding them through data preprocessing, model configuration, training, and evaluation to implement such an NLI model for factuality assessment.



## Getting Started
---

### Prerequisites

- Access to [Google Colab](https://colab.research.google.com/).

### Installation

1. Open the notebook in Google Colab.
2. Run the initial cells to:
   - Check the GPU type. (A100 perferred)
   - Install necessary libraries (`transformers`, `datasets`, etc.).
   - Mount Google Drive (if you're using it for data storage).

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

### **Model Training Loop**
---

#### Initialization:
- The model is set to training mode with `nli_model.train()`.
- The AdamW optimizer is initialized with specified learning rate and weight decay.
- A `GradScaler` is initiated for Automatic Mixed Precision (AMP) to speed up training and reduce memory usage without compromising the model's accuracy.
- Early stopping parameters are set up to prevent overfitting and save computational resources.
- A learning rate scheduler (`ReduceLROnPlateau`) is used to adjust the learning rate based on the validation loss, helping in achieving faster convergence.
- Lists `train_losses` and `valid_losses` are initialized to keep track of losses for visualization and analysis.

#### Epoch Loop:
- For each epoch, the model processes batches of data from the training loader.
- The optimizer's gradients are zeroed out at the start of each batch to prevent accumulation.
- The forward pass is performed with autocasting, suitable for AMP.
- Only the logits corresponding to `entailment` and `contradiction` are considered, ignoring `neutral`.
- The cross-entropy loss is computed with class weights to handle any class imbalance.
- Backpropagation is done with the scaled loss, a part of the AMP process.
- The optimizer's parameters are updated.
- Periodically (every 100 data points or `100/bs` batches), the running training loss is printed, and validation loss is computed using a separate function (`compute_validation_loss`). Both losses are stored for future visualization.
- The learning rate scheduler adjusts the learning rate based on the validation loss.
- If the validation loss is the lowest seen so far, the current model state is saved as the best model.

#### Early Stopping:
- After each epoch, the average validation loss is checked against the previous minimum average validation loss.
- If the average validation loss does not decrease for a specified number of epochs (`early_stopping_limit`), the training is halted to prevent overfitting. The best model state, saved during training, is loaded back into the model.

### **Model Testing Loop**
---

#### Model Evaluation Mode:
- The model is set to evaluation mode with `nli_model.eval()`, ensuring that certain layers like dropout are disabled during the testing phase.

#### Initialization:
- The total number of batches in the test loader is determined.
- The `test_loss` variable is initialized to accumulate the loss over the test set.
- Lists `test`, `prob`, and `pred` are initialized to store true labels, predicted probabilities, and predicted labels, respectively.

#### Evaluation Loop:
- A loop runs through each batch in the test loader.
- The forward pass is performed on the batch.
- Logits corresponding to `entailment` and `contradiction` are extracted, excluding `neutral`.
- Cross-entropy loss is computed for the batch and accumulated in `test_loss`.
- The logits are normalized using the softmax function.
- The probability of predicting the label as 1 (i.e., `contradiction`) is extracted and added to the `prob` list.
- The predicted labels are determined by taking the argmax of the logits and are added to the `pred` list.
- True labels from the batch are added to the `test` list.

#### Results:
- The average test loss is computed by dividing the accumulated `test_loss` by the number of batches.
- The true labels (`test`), predicted probabilities (`prob`), and predicted labels (`pred`) are printed for analysis.
- The average test loss is displayed.

### **Results & Visualization**
---

#### Loss Visualization:
- A plot is generated which showcases the training and validation loss over iterations.
- This visualization aids in understanding the convergence of the model and highlighting if any overfitting or underfitting occurred.

#### Evaluation Metrics:
- A confusion matrix is printed, offering a detailed view of the true positive, false positive, true negative, and false negative counts.
- Key classification metrics are computed and displayed:
  - **Accuracy**: The ratio of correctly predicted samples to the total samples.
  - **Recall**: The ratio of correctly predicted positive samples to the actual positives.
  - **Precision**: The ratio of correctly predicted positive samples to the total predicted positives.
  - **F1 Score**: The harmonic mean of precision and recall.
  - **AUC (Area Under the Curve)**: Represents the model's ability to distinguish between classes, with a focus on the ROC curve for this binary classification.

#### Classification Report:
- A detailed report is printed, which provides metrics like precision, recall, and F1-score for each class. This offers an in-depth view of the model's performance on a class-wise basis.

#### Precision-Recall Curve:
- A precision-recall curve is plotted. This visualization is crucial for understanding the trade-off between precision and recall for different threshold values, especially useful in the context of imbalanced datasets.
- The Area Under the Precision-Recall Curve (PR AUC) is also computed and displayed, providing a singular metric to gauge the model's performance in terms of precision and recall.

### Troubleshooting
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


### Limitations
---
1. **Token Limitation**:
   - As seen in the `FeedbackDataset` class:
     ```python
     class FeedbackDataset(Dataset):  # DataFrame
         def __init__(self, data):
             self.model_inputs = tokenizer(data['doc'].tolist(), data['feedback'].tolist(),
                                           max_length=1024, truncation='only_first', padding='max_length')
     ```
     The model has a token limit of 1024. If the 'doc' or hypothesis exceeds this token count, it will be truncated, leading to potential loss of information. This limitation arises from the underlying architecture of the model and its handling of token sequences.

2. **Computational Complexity**:
   - The model is computationally intensive. As a result, the training process can be time-consuming, especially on datasets with large volumes or complexity. Additionally, the GPU usage during training and evaluation is considerably high, which can lead to memory-related issues if not managed properly.

