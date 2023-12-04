# sale_prediction_self_attention_and_classical_models

The main goal of this project is to build a simple deep network using multi-head self-attention mechanism for time series prediction and compare its performance with classical models, e.g decision tree, XGBoost. Note that I use attention mechanism because it is explainble. For dataset, we use the Walmart Dataset from Kaggle. Here is the architecture of my deep network:
- Positional encoding layer: attention mechanism does not know the positions, and we have to use positional encoding.
- Multi-head self-attention layer: This layer tells us about how the current input depends on previous input.
- Normalization layer with residual connection: We use residual connection to reduce possible noises/gradient vanishing and then normalize.
- Linear layer: this is for producing the predicted sale values.

The performance of this deep network on this dataset is not impressive, and it is unstable. There are several points I need to consider further:
- Is the architecture too simple?
- How to capture the relationship among data better after the self-attention layer?
- Is the dataset too small/too simple?

Any (partial) answer of these questions and further discussion on how to use attention mechanism for time series prediction effectively are more than welcome. Thanks for reading!