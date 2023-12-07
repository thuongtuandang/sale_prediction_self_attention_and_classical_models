# sale_prediction_self_attention_and_classical_models

The main goal of this project is to build a simple deep network using multi-head self-attention mechanism for time series prediction. Note that I use attention mechanism because it is explainble. For dataset, we use the Walmart Dataset from Kaggle. In the notebook file, I also explored the data and use tree-based models for the prediction. Here is the architecture of my deep network:

- Positional encoding layer: attention mechanism does not know the positions, and we have to use positional encoding.
- Multi-head self-attention layer: This layer tells us about how the current input depends on previous inputs.
- Normalization layer with residual connection: We use residual connection to reduce possible noises/gradient vanishing and then normalize.
- Linear layer: this is for producing the predicted sale values.

Here is how I preprocess the data: we will use the sales of previous weeks to predict the sale of the next week. You can change this number in the codes, I set it as input_chunk = 4, i.e. we will use the sales of 4 previous week to predict the sales of next week. Note that this is a multiple time series prediction, so a pair (X, y) must belong to the same store. After that, we can divide (X, y) to train and test set and shuffle them.

The performance is not impressive, but it is acceptable. You can see the training in the file training_steps.png. Best result so far is RMSE loss for training set around 280k and for test set 430k (see two files results.png and plotting_430k_lost.png), because of the following reasons:

- The size of dataset is quite small.
- It is the multiple time series task and each time series has only 143 rows.
- I haven't used some additional features, for example: holiday or the weather of next week.
- The processing time is slow, so I don't use many loops to adjust the input_chunk and the learning rate.
- I didn't use GPU to speed up the training.

Comments and suggestions are welcome. Thanks for reading!