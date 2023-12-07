# sale_prediction_self_attention_and_classical_models

The main goal of this project is to build a simple deep network using multi-head self-attention mechanism for time series prediction. For dataset, we use the Walmart Dataset from Kaggle. In the notebook file, I explored the data and use tree-based models for the prediction. Note that I use attention mechanism because it is explainble. Here is the architecture of my deep network:

- Positional encoding layer: attention mechanism does not know the positions, and we have to use positional encoding.
- Multi-head self-attention layer: This layer tells us about how the current input depends on previous inputs.
- Normalization layer with residual connection: We use residual connection to reduce possible noises/gradient vanishing and then normalize.
- Linear layer: this is for producing the predicted sale values.

Because it is a multiple time series problem, I divide them into prediction for a certain store. In codes, I predict the current weekly sales of store 5 (you can change it in the file experiments.py) according to the sales of the 4 previous weeks. The loss for the test set is around 21k, but we cannot conclude anything, because if there is another time series of another stores, the attention weights will change, and it is hard to tell. For such a small data set, I think it is not impressive compared to tree-based models. I might need to try it with larger datasets, or add an additional layer for cross-attention.