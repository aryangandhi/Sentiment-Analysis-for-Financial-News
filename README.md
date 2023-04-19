# CSC413_final_project


## Introduction

We have created a deep learning model for sentiment analysis on financial news using a multilayer perceptron (MLP) architecture with dropout layers. The goal of this model is to classify the sentiment expressed in financial news articles as either “positive”, “negative”, or “neutral”. This is a supervised learning task where the model is trained on labeled data containing financial news text and their corresponding sentiment labels.

The input to the model is a representation of the financial news text, typically a fixed-length vector obtained by processing the text using natural language processing techniques like tokenization, embedding, or pre-trained language models. In our case, a pre-trained BERT model is used to obtain the embedded text representations. The output of the model is a 3-dimensional vector representing the estimated probability for each of the three sentiment classes: “positive”, “negative”, “neutral”. The model predicts the class with the highest probability value as the sentiment label for the given input text.

Our model incorporates ReLU activation functions and dropout layers for regularization, which helps prevent overfitting during training. The architecture consists of multiple fully connected layers that transform the input text representation into the final output. Additionally, the model is trained using the cross-entropy loss function.

<br>

## Model Figure
![ModelArchitectureDiagram](https://user-images.githubusercontent.com/43526001/232604630-dbcc72ed-33e3-4f63-9ab8-2ea1155de2c5.jpg)


<br>

## Model Parameters
Since our model is an MLP model with 5 fully connected layers,

total # of parameters in the model  =  
\# of parameters in the first layer + # of parameters in the second layer + # of parameters in the third layer + # of parameters in the fourth layer + # of parameters in the fifth layer

Number of parameters in each layer:


- \# of parameters in the first layer: 768 * 256 (weights) + 256 (biases) = 196,864
- \# of parameters in the second layer: 256 * 128 (weights) + 128 (biases) = 32,896
- \# of parameter in the third layer: 128 * 64 (weights) + 64 (biases) = 8,256
- \# of parameters in the fourth layer:  64 * 32 (weights) + 32 (biases) = 2,080
- \# of parameters in the fifth layer: 32 * 3 (weights) + 3 (biases) = 99

Thus,

Total # of parameters = 196,864 + 32,896 + 8,256 + 2,080 + 99 = 240,195

The parameters in the model come from the weights and biases of each fully connected layer. During training, these parameters are updated to minimize the loss function for the given task of performing sentiment analysis on financial news data. Although the model includes dropout layers, the dropout layers themselves do not have any learnable parameters. They are used to prevent overfitting by randomly setting 50% of the input units to zero during training.

<br>

## Model Examples

### Correctly Classified Text
$\bf{Text:}$ “efficiency improvement measures 20 January 2010 - Finnish stationery and gift retailer Tiimari HEL : TII1V said today that it will continue to improve its operational efficiency , by focusing on its profitable core operations .”

$\bf{Prediction:}$ positive

$\bf{Label:}$ positive.

### Incorrectly Classified Text
$\bf{Text:}$ “The company also said that the deployment of the Danish 4G network continues and it expects to cover 75 % of the Danish population in 2011 .”

$\bf{Prediction:}$ positive

$\bf{Label:}$ neutral.

<br>

## Data Source
We are gathering our data from kaggle, which is an online community platform, that contains free-to-use data.Here is the link of the data we intend to use: [Sentiment Analysis for Financial News | Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)

Citation of the data: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014).

<br>

## Data Summary
There are 4,846 total rows of data that we partition into training, validation and test sets. This is a significantly larger amount of data compared to what we used in A1 and A2 (roughly 1000 for each),. Thus, we believe the amount is sufficient to train and test and model.

#### Number of examples in each sentiment class:

$\bf{neutral:}$ 2879

$\bf{positive:}$ 1363

$\bf{negative:}$ 604

Although the class distribution is not uniform we can see that we have enough examples of each type of label in order to successfully train, validate and test our model.

We have noted that there is a space between punctuation, such as periods and commas, presumably for our benefit (considering these as words, then splitting the sentence using an empty space as a delimiter).

![sentences_pic](https://user-images.githubusercontent.com/43526001/232619145-4571db3a-d481-4488-984b-9106d245dc15.png)

We can also note that the average length of a sentence is 23 words and the number of total words is 111906.

![sentence_length_pic](https://user-images.githubusercontent.com/43526001/232619337-b4573870-695a-4007-90ef-c42b42d4cc8d.png)

Lastly, we can see that the number of unique words is 12966. The reason this is so high is because the model will have to consider each different occurrence of a float a “word”. Additionally, in the below image we can view the top 10 most common words that appear in the data set.

![most_common_words_pic](https://user-images.githubusercontent.com/43526001/232619632-8eedb7d3-2e70-4386-97fc-35e91cfcf3b3.png)



## Data Transformation
We started by taking the google sheet file, and reading it into a data frame. The data consists of two columns, the text and the label. We marked the column with the text as ‘x’, and the column with the label as ‘t’. We then converted the dataframes to numpy arrays, as they are easier to work with for things such as splitting data. At this point, we attempted to try some data augmentation techniques. We wanted to use data augmentation to expose the model to a variety of language patterns, increasing its ability to handle different types of input data. The first technique we tried was appending a sentence to the initial training data for a sentence that contained synonyms for each word. We found that this heavily impacted the data quality (for example, the algorithm changed the word ‘move’ to ‘make_a_motion’). This is troublesome as it increases our vocabulary (with words that are not in the dictionary), as well as creates sentences that do not make sense (it’s unlikely that anyone would construct sentences the way the sentence_to_synonym library does). Another strategy we implored was a paraphrasing the text. However, this proved to be very complicated; the paraphrasers we used would either just reduce the number of sentences (keeping the first sentence), or it would commonly return a boolean if only one sentence needed to be paraphrased (would output whether or not the sentence was true or not). Lastly, we settled on translating the training data to French, then translating it back to english. We took the text from the training data, then stored the augmented data in a pickle file. The pickle file was then appended to the training set. Below is an example of a sentence found in the training data, then the augmented version of the text (this can be found on the output of the colab file, under Data Augmentation):

#### Example of Data Augmentation
$\bf{Original Text:}$ "Exel Composites ' long-term growth prospects remain favourable , however ."

$\bf{Augmented Text:}$ "However, Exel Composites 's long-term growth outlook remains favourable ."

<br>

## Data Split

We split the data into a 60/20/20 partition. We felt as though we didn’t need to account for other factors (the timing of the financial news, for instance) since these types of factors won’t negatively impact our training. For example, if the training data consisted of only financial news from this year, and the validation/test data was from last year, this would not be an issue. In addition, the data did not contain any other information (person issuing the statement, time of statement etc.), so we wouldn’t be able to make any adjustments anyways. We used 60/20/20 as this is considered a standard split in practice. 

## Training Curve

#### Loss Curve
<img width="328" alt="Loss Curve" src="https://user-images.githubusercontent.com/43526001/232613825-68a9e00a-de89-4103-95d1-b2179b59977e.png">

#### Accuracy Curve
<img width="320" alt="Accuracy Curve" src="https://user-images.githubusercontent.com/43526001/232614090-bbf25069-6a42-4e6a-b078-02a893d715ac.png">

<br>

## Hyperparameter Tuning
For hyper parameter tuning, we tried hyper parameterizing three models: regular MLP, regular CNN, and a dropout layer MLP. As for the type of tokens we tried, we tried various features like the BERT tokens/features and the GPT2 tokens/features. We also tried two different training functions where one was the regular training style in assignment one and assignment two with Adam gradient descent, while the second training style involved gradient clipping. We wanted to see if gradient clipping was efficient with models other than RNN’s. Lastly we tuned the batchsize, weight decay, and the learning rate. For batchsize, we tried sizes 16, 32, and 64. For learning rate, we tried values 0.0005, 0.001, and 0.005. Lastly for weight decay we tried values 0, 0.005, and 0.01.

<br>

## Quantitative Measures
For the quantitative measures, we measured the percentage that our model would classify a review as either positive or negative. We calculated the percentage by counting all the reviews our model guessed correctly and divided it by our total amount of reviews. Likewise, this was done for the percentage of incorrect reviews. Using a training data of size 5814, we acquired a training percentage of 88.30%. This means 5133 financial reviews were correctly classified while 681 were not classified correctly. As for the validation data, the data size we used was a size of 969 while achieving a validation accuracy of 79.77%. This means that 772 of the financial reviews were correctly classified while 197 were classified incorrectly. Another quantitative measure we used was to depict which model we should use. To measure this, we simply trained the different models using the same hyper parameters, data set, and the same method of training (gradient descent using the adam optimizer). We then chose our final model based on the model yielding the highest validation accuracy. 


## Quantitative and Qualitative Results

### Quantitative Results
The model can be evaluated by the accuracy on the test set which yielded a value of 78.56%. Additionally, the final training accuracy is 87.74% and the final validation accuracy is 79.88% with a final training loss of roughly between 0.004 and 0.006.  

We can visually inspect how our model performed in the test set using a confusion matrix:

<img width="356" alt="Confusion Matrix" src="https://user-images.githubusercontent.com/43526001/232617369-ed8eef00-5ab2-44f8-adcc-00543f9cb7d2.png">

The confusion matrix visually informs us the percentage of correctly and incorrectly classified test cases for each sentiment class. The values along the diagonal of the matrix represent the percentage of cases where the model correctly classified the sentiment of the text. Based on the confusion matrix, the sentiment class that is classified correctly the most are neutral texts with a correct classification rate of 89.98%, while negative and positive texts are correctly classified at a rate of 81.25% and 55.36%, respectively. These differences in accuracy rate among the classes is likely explained by the difference in the number of examples per class in the dataset. In the overall dataset there are 2879 texts with neutral sentiment, 1363 with positive sentiment, and 604 with negative sentiment. Thus, this imbalance in training examples among the classes is likely responsible for the discrepancy in the classification accuracy. To mitigate this issue, more text with positive and negative sentiment should be added to the dataset.

### Qualitative Results
When we inspect some specific examples of what data the model classified correctly and what it misclassified, we can make a few observations and infer possible patterns in the model’s errors. We see that examples of text that convey facts are likely to be misclassified as the model is not able to properly identify which facts are positive and which are negative. We also see that when positive or negative sentiments are misclassified, they are more likely to be misclassified as neutral than as the opposite sentiment (e.i: positive sentiment being classified as negative or negative sentiment being classified as positive is not as likely). Whereas, a neutral statement has a pretty good chance of being misclassified as either positive or negative.  This may be due to the fact that neutral statements make up a large percentage of our overall dataset that we train on. This indicates our model likely does not understand nuances in language very well as it has not had as many examples of texts that contain positive or negative sentiments, which would have more subtle complexities for the model to learn. Additionally, financial data presents a unique challenge in sentiment analysis due to the presence of specific keywords and jargon that can be ambiguous or hard to classify as either positive or negative. The complexity arises from the fact that the meaning and sentiment behind these keywords often depend on the context in which they are used. For example, the word “profit” could generally be viewed with positive sentiment and “debt” could generally be viewed with negative sentiment. However, if the sentence is “there was a decline in profit” or “there was a decline in debt”, the sentiments would be negative and positive respectively, despite some keywords having the opposite sentiment. In order to counteract this issue, additional training data and further data augmentation on text data with positive and negative sentiments may be required. 

## Justification Results
The implemented method, a multilayer perceptron (MLP) model with dropout layers, performed reasonably well considering the difficulty of the sentiment analysis problem. The model achieved a training accuracy of 87.74%, a validation accuracy of 79.88% and a test accuracy of 78.56%. Given that sentiment analysis for financial data can be a difficult task, the reported accuracies demonstrate the model performed fairly well.

### Initial Hypotheses

- One of the hypotheses we had was to use gradient clipping and using a learning rate scheduler to lower the oscillation in the loss function and have smoother convergence. Unfortunately, we found that the accuracy of our validation and test set was significantly lower using gradient clipping and learning rate scheduler no matter how we modified the parameters and hyperparameters. 
- We also hypothesized whether using transfer learning with GPT-based features as input might yield better performance compared to BERT-based features. However, our experiments demonstrated that this approach was less accurate across all three of our model architectures. This indicates that BERT-based features may be more suitable for the task of sentiment analysis in our specific context.
- Our last hypothesis was to see which model was performed the best: a regular MLP, a CNN, or a modified MLP that has a dropout layer after each other layer. The MLP with dropout layers had the best performance on the validation set, likely due to its ability to use regularization to prevent overfitting.

### Why our model is reasonable
1. Data quality: The dataset used for training, validation, and testing consisted of 4,846 financial news text samples with a reasonable distribution among the three sentiment classes: 2,879 neutral, 1,363 positive, and 604 negative. The rate for successful classification may be higher for neutral statements due to the large number of neutral examples in our dataset. However, the distribution we get in the end is reasonable for training a model that distinguishes between three sentiment classes.
2. Pre-processing and representation: The pre-trained BERT model facilitated the learning process of our MLP model as it provided an effective way of capturing the information important for understanding the sentiment in texts containing financial news. By representing text as the features of the pre-trained BERT model, the MLP can identify patterns in the data and the relationships within the text more effectively.
3. Architecture and regularization: The MLP model was designed with 5 fully connected layers, ReLU activation functions, and dropout layers for regularization. This architecture helped to prevent overfitting during training and allowed the model to generalize well to new data. We compared multiple model architectures and trained them using different numbers of hidden layers and units and the architecture that we found had the highest validation accuracy was chosen as the final model.
4. Hyperparameter tuning: The model underwent an extensive hyperparameter tuning process, which included testing different model structures (MLP, CNN), tokenization and transfer learning methods (BERT, GPT-2), training functions (gradient descent with Adam optimizer, gradient clipping, learning rate scheduling), batch sizes, learning rates, and weight decay values. This thorough optimization process contributed to the model's overall performance.

Given the inherent difficulty of the sentiment analysis problem, especially in the domain of financial news where language can be complex and nuanced, the achieved results are reasonable. However, there is still room for improvement, particularly in the model's ability to correctly classify positive and negative sentiment. To enhance the model's performance further, we could collect more training data, particularly focusing on increasing the number of positive and negative sentiment examples in order to balance out the class distribution.  Finally, while our model is robus different sentence structures due our use of data augmentation, we could make the model even better at correctly classifying text with positive and negative sentiment through further data augmentation. 

## Ethical Considerations
- Financial data contains sensitive personal information  and must be careful to adhere to data privacy laws.
- The outcome of predictions can have a large effect on a company, so it is important that we don’t misrepresent the success of a company. For example, misconstruing neutral news for negative news could cause shareholders to sell shares of a public company, which would obviously cause the company’s stock price to drop.
- This model checks for the sentiment of the information provided based on the text; however, text being positive should not be misinterpreted as “good”. Any fact can be presented as good news or bad news depending on the bias/perspective of the person/company that authored the information. 
- The perspective of the news is from a retail investor. Meaning, good news for an investor could be poor news from another party (a company or competitors of a company, for example).

## Authors

$\bf{Aidan:}$ Discovered the data source, performed the data transformation (including data augmentation), and performed the data exploration. Also added the ethical considerations, and assisted in the justification of results.

$\bf{Aryan:}$ Preprocessed the data, created MLP and DropoutMLP model architectures, set up training process (created train, train2 and get_accuracy functions). Also completed model parameters, model examples, quantitative results and assisted in the justification of results.

$\bf{Pushti:}$ Created model figure, CNN model architecture, and did an analysis on performance of the model in the test set. Also completed the introduction, qualitative results, and assisted in the justification of results.

$\bf{Arlyn:}$ Completed hyperparameter tuning and assisted in data augmentation. Also completed the quantitative measurements, assisted in training curve and justification of results.
