# Custom Sentiment Classification

## Halil İbrahim Dönmezbilek

### Dataset
I start by splitting my train dataset into train and validation datasets (80/20 ratio). In my exploratory analysis, I also try to understand my validation dataset. The distribution of the dataset is imbalanced for train and validation but balanced for the test dataset. The number of positive results is very low, so my model will probably suffer from predicting positive labels. Also, I try to understand the nature of the conversation by looking at which areas are more fed to the dataset and especially order-related conversations that are held. Most of the conversations are less complex, so the agent’s conversation may include noise to model. In hypertuning, one of the things I am going to change is I will try to use only customer conversations with the model. The training dataset contains junior experience agents. I assume a junior employee can help a customer much longer than an experienced employee. In other words, juniors can talk too much compared to experienced people. This is another reason for eliminating the agent conversation at the tuning stage.

#### These are my preprocess steps:
- **CustomerText Filtering**:
  - I have mentioned in the EDA part that this step filters only customer text. I will rerun these steps at the beginning of the hyperparameter tuning stage.
- **Case Normalization**:
  - All text was converted to lowercase to maintain uniformity and prevent the same words in different cases from being treated as different.
- **Pattern Removal**:
  - Standard introductory phrases used by agents (e.g., "Thank you for calling BrownBox Customer Support. My name is [Agent Name]. How may I assist you today?") were removed to reduce noise and focus on the customer's language.
- **Cleaning Role Identifiers**:
  - Role identifiers such as "Agent:" and "Customer:" were stripped from the text to prevent any noise and clean the text.
- **Removing Emails and Websites**:
  - I assume any Email addresses and URLs do not have any valuable information for sentiment classification.
- **Punctuation and Numeric Removal**:
  - All punctuation marks and numeric characters were removed to focus purely on textual data. This helps in reducing the complexity of the model's vocabulary.
- **Whitespace and Repetition Normalization**:
  - Extra spaces and repeated punctuation were removed to clean up the text. This standardization helps in maintaining a consistent data format.
- **Emoji Conversion**:
  - Since Emojis have a semantic meaning but are not expressed with language, any Emojis were translated into corresponding text descriptions to include their emotional content in the text analysis.
- **Special Character Removal**:
  - Ensuring special characters were stripped to focus the analysis on alphanumeric characters and standard whitespace.
- **Expansion of Contractions**:
  - Contractions were expanded to their full forms (e.g., "can't" to "cannot") to standardize text and improve the model's understanding.
- **Stopword Removal**:
  - Common English stopwords were removed to focus on the more meaningful words in the text, which are likely to contribute more to sentiment analysis.
- **Lemmatization**:
  - Words were lemmatized to their base forms to reduce the inflectional forms presented to the model, thus simplifying the model's task of understanding the text.

### Modelling
In the original [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT), the attention mechanism was casual self-attention. For sentiment analysis, my dataset should not be masked; all the data should be red. FullSelfAttention allows each position in the decoder to attend to all positions in the input sequence. This complete contextual visibility can help in better understanding the sentiment in the text. Thus, I close the mask part and transform it into a self-attention mechanism. This class is assigned the FullSelfAttention.

Since I have changed the attention mechanism in the model, I have to redefine the Block so I have a BumbleBeeBlock class that is inherited from `GPT.Block`. I will use this block to redefine the GPT model for sentiment classification, BumbleBee. In the BumbleBee class, it is inherited from GPT, and I have changed some functions by overriding the variables. My first override is defining the new block using BumbleBeeBlock, and the second override is changing the last layer of GPT (`lm_head`) for sentiment classification dimensions which are positive, neutral, and negative. My second override is a forward function to recalculate the logits on the new lm_head layer. My third override is the `from_pretrained` method to ensure the pre-trained model gpt2 will align with BumbleBee. Since I have changed the last layer of the model, pre-trained weights should be assigned to BumbleBee except for lm_head. The model will learn the lm_head layer.

### Evaluation
I will use f1 macro, precision, recall, and confusion matrix as evaluation metrics. I will select the best f1 macro-scored model for testing purposes. The F1 Score (Macro) gives a balanced measure of the model's precision and recall together. Precision and recall show how the model makes mistakes (like false positives and false negatives). The confusion matrix provides a clear picture of how well the model performs for each class and shows where it might be getting things wrong. This will help to identify where the model needs improvement.

### Results
<table style="width:100%;">
  <tr>
    <td style="width:33.33%; display:flex; justify-content:center; align-items:center;"><img src="https://github.com/halildonmezbilek/CustomSentimentClassification/assets/40296559/8f999cbc-6472-42ce-a7b7-ad4e0cb2aaa0" alt="test_f1_macro" style="max-width:100%;"></td>
    <td style="width:33.33%; display:flex; justify-content:center; align-items:center;"><img src="https://github.com/halildonmezbilek/CustomSentimentClassification/assets/40296559/8dfd6c9e-ad79-4f26-862a-ca63b2dc197f" alt="train_f1_macro" style="max-width:100%;"></td>
    <td style="width:33.33%; display:flex; justify-content:center; align-items:center;"><img src="https://github.com/halildonmezbilek/CustomSentimentClassification/assets/40296559/4cea22d0-0dfa-4f2b-8ad7-2fc92f89ec1e" alt="val_f1_macro" style="max-width:100%;"></td>
  </tr>
  <tr>
    <td style="display:flex; justify-content:center; align-items:center;"><img src="https://github.com/halildonmezbilek/CustomSentimentClassification/assets/40296559/30ddc07a-34e5-4894-bc88-ace935623544" alt="test_loss" style="max-width:100%;"></td>
    <td style="display:flex; justify-content:center; align-items:center;"><img src="https://github.com/halildonmezbilek/CustomSentimentClassification/assets/40296559/1a3b4ac6-c670-45ee-87c9-10e50aa70db2" alt="train_loss" style="max-width:100%;"></td>
    <td style="display:flex; justify-content:center; align-items:center;"><img src="https://github.com/halildonmezbilek/CustomSentimentClassification/assets/40296559/ff02892e-3f5e-4023-b01d-64e341e77dec" alt="val_loss" style="max-width:100%;"></td>
  </tr>
  <tr>
    <td style="display:flex; justify-content:center; align-items:center;"><img src="https://github.com/halildonmezbilek/CustomSentimentClassification/assets/40296559/5d59ec9f-d5c3-4314-ada0-7d716d16c882" alt="test_precision_macro" style="max-width:100%;"></td>
    <td style="display:flex; justify-content:center; align-items:center;"><img src="https://github.com/halildonmezbilek/CustomSentimentClassification/assets/40296559/cbf25935-9d40-42d5-986d-6bad67a243a5" alt="train_precision_macro" style="max-width:100%;"></td>
    <td style="display:flex; justify-content:center; align-items:center;"><img src="https://github.com/halildonmezbilek/CustomSentimentClassification/assets/40296559/e4b79663-d23d-4573-a290-03142f94c0ab" alt="val_precision_macro" style="max-width:100%;"></td>
  </tr>
  <tr>
    <td style="display:flex; justify-content:center; align-items:center;"><img src="https://github.com/halildonmezbilek/CustomSentimentClassification/assets/40296559/1ff21405-1b7a-4e7f-8caa-70e688b9f261" alt="test_recall_macro" style="max-width:100%;"></td>
    <td style="display:flex; justify-content:center; align-items:center;"><img src="https://github.com/halildonmezbilek/CustomSentimentClassification/assets/40296559/2f31e7c3-9a6e-47cd-b1a2-431e7eca2e82" alt="train_recall_macro" style="max-width:100%;"></td>
    <td style="display:flex; justify-content:center; align-items:center;"><img src="https://github.com/halildonmezbilek/CustomSentimentClassification/assets/40296559/aeb59196-1d3d-4f51-9e96-f9765ff4fa01" alt="val_recall_macro" style="max-width:100%;"></td>
  </tr>
  <tr>
    <td colspan="2" style="width:50%; display:flex; justify-content:center; align-items:center;"><img src="https://github.com/halildonmezbilek/CustomSentimentClassification/assets/40296559/b3e5dd3c-0db9-444a-b56c-6e651b8ffb4f" alt="train_accuracy" style="max-width:50%;"></td>
    <td style="width:50%; display:flex; justify-content:center; align-items:center;"><img src="https://github.com/halildonmezbilek/CustomSentimentClassification/assets/40296559/09d4467f-1e7b-4e2f-9eb4-7bae0bbd91f7" alt="val_accuracy" style="max-width:100%;"></td>
  </tr>
</table>

<table style="width:100%;">
  <tr>
    <td style="display:flex; justify-content:center; align-items:center;"><img src="https://github.com/halildonmezbilek/CustomSentimentClassification/assets/40296559/60e759a5-bb6a-4859-b0c1-9a0e8ed0ef66" alt="confusion-matrix-scratch" style="max-width:100%;" ></td>
    <td style="display:flex; justify-content:center; align-items:center;"><img src="https://github.com/halildonmezbilek/CustomSentimentClassification/assets/40296559/60f4d749-c63f-4395-80d4-97fe06c7085d" alt="confusion-matrix-fine" style="max-width:100%;" ></td>
    <td style="display:flex; justify-content:center; align-items:center;"><img src="https://github.com/halildonmezbilek/CustomSentimentClassification/assets/40296559/7f61af01-aac4-4d1a-bbeb-02fa2dd3941e" alt="confusion-matrix-hyper" style="max-width:100%;" ></td>
  </tr>
  <tr>
    <td>From Scratch</td>
    <td>Fine Tunning</td>
    <td>Hyperparameter Tunning Best Model</td>
  </tr>
</table>







These results with an interactive report are available with this link:  
[Custom Sentiment Classification Results](https://wandb.ai/halil-donmezbilek/Custom-Sentiment-Classification/reports/Custom-Sentiment-Classification-Results--Vmlldzo4NTAyNjMx)

### Discussion
|                               | F1 Macro | Accuracy |
|-------------------------------|----------|----------|
| **Scratch Model**             | 48.12%   | 53%      |
| **Fine Tuning**               | 52.41%   | 57%      |
| **Hyperparameter Tuning (Best Model)** | 74.52%   | 77%      |



My fine-tuned model (second model) has a higher f1 macro and accuracy than the scratch model (first model). For the first two models, I have used the same tokenizer for training. GPT2 has pretrained 125M parameters and scratch model 30M parameters. The reason for the difference in performance metrics is that the second model has a much deeper network; however, it does not increase the performance metric while the network is deeper.

When I observe these two models, they tend to memorize the train data. In the hyperparameter tunning part, I have selected some regulator parameters to generalize the model. These are the parameters that I select for hyperparameter tuning: learning_rate = [0.001, 0.0001], weight_decay[0.1, 1.5], max_grad_norm [0.5, 1.0]. I chose the learning rate to carefully control how quickly the model updates itself, aiming to achieve steady and general learning without rushing, which helps prevent overfitting. I choose Weight Decay to enforce a simplicity constraint on the model’s structure. I choose the maximum gradient norm to ensure that training updates are moderate and consistent, preventing drastic changes in model weights that could destabilize the learning process. Also, in the hyperparameter tunning step, I have changed the tokenizer to BERT, and as I mentioned before in the EDA process, I only use the customer context. At the end of the hyperparameter tuning, the best model gives 77% accuracy. When examining the confusion matrix, it is evident that the model significantly improves prediction accuracy for neutral labels and shows considerable enhancement in identifying positive labels. However, it performs better in predicting neutral labels than positive labels.

### Conclusion
This project shows how well our sentiment analysis model works by making smart changes and carefully choosing settings during training. I came up with a modified model called BumbleBee, which better understands feelings in text. Hyperparameter changes helped the model not just memorize the training data but actually learn from it, leading to 77% accuracy.

[WANDB Project](https://wandb.ai/halil-donmezbilek/Custom-Sentiment-Classification?nw=nwuserhalildonmezbilek)
