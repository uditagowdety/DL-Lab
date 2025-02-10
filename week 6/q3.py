# #pytorch checkpoints
#
# Initialize Model, Optimizer, Loss Function
#     Define model, optimizer, loss function.
#
# Load Checkpoint (if exists)
#     If checkpoint exists:
#         Load model.state_dict(), optimizer.state_dict(), epoch, and loss.
#     If no checkpoint:
#         Start fresh from epoch 0.
#
# Training Loop
#     For each epoch:
#         Iterate over batches:
#             Forward pass → compute loss → backward pass → update weights.
#         Track loss, accuracy.
#
# Monitor Performance
#     After each epoch, evaluate on validation set.
#     If performance improves (e.g., accuracy), save checkpoint.
#
# Save Checkpoint
#     Save model.state_dict(), optimizer.state_dict(), epoch, and performance metrics if better than previous best.
#
# Resume Training (if needed)
#     If resuming, load checkpoint (model.state_dict(), optimizer.state_dict(), epoch).