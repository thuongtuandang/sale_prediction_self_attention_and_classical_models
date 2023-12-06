# sale_prediction_self_attention_and_classical_models

The main goal of this project is to build a simple deep network using multi-head self-attention mechanism for time series prediction and compare its performance with classical models, e.g decision tree, XGBoost. Note that I use attention mechanism because it is explainble. For dataset, we use the Walmart Dataset from Kaggle. Here is the architecture of my deep network:

- Positional encoding layer: attention mechanism does not know the positions, and we have to use positional encoding.
- Multi-head self-attention layer: This layer tells us about how the current input depends on previous inputs.
- Normalization layer with residual connection: We use residual connection to reduce possible noises/gradient vanishing and then normalize.
- Linear layer: this is for producing the predicted sale values.

Also, because it is a multiple time series problem, I divide them into prediction for a certain store. In codes, I predict the current weekly sales of store 1 according to the sales of the 4 previous week. Best score so far is 95000 RMSE loss for one store (See file RMSE_loss_store_1.png). Let me convert this into python codes for 45 stores.

# Python codes
RMSE_store_1 = 95000
num_rows_store_1 = 150
num_stores = 45
total_rows = 6500

# Calculate MSE and SE losses
MSE = RMSE_store_1 ** 2
SE_loss_store_1 = MSE * num_rows_store_1
SE_loss_total = SE_loss_store_1 * num_stores

# Calculate total MSE and RMSE across all stores
MSE_total = SE_loss_total / total_rows
RMSE_total = (MSE_total) ** 0.5

print(f"Total RMSE loss across all stores: {RMSE_total:.2f}")

# Result: Total RMSE loss across all stores: 96809.69

It is quite good, because the RMSE loss for XGBoost is 87000.