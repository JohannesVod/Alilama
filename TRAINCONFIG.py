# Model Parameters:
class ModelParams:
    model_name:str = "alilama"
    d_model:int = 480 # embedding dimension
    blocks:int = 8 # number of transformer blocks that are repeated
    max_seq_len:int = 128 # sequence length the transformer is trained on
    num_heads:int = 8 # heads in the multiheaded attention
    hidden_dim:int = 4*d_model # size of linear transformation inside MLP
    head_width:int = d_model//num_heads # width of head
    assert head_width * num_heads == d_model

# Training Parameters
BATCH_SIZE = 128
learning_rate = 5e-5
train_steps = 1000000 # Numbers of batches used for training
beta1 = 0.9 # Parameter for adam optimizer
beta2 = 0.95
weight_decay = 0 # used to reduce overfitting
compile_model = False # compiles the model but takes some time

# Logging:
log_interval = 5 # interval where we print the loss function
eval_interval = log_interval*20 # interval where we evaluate the model
eval_batches = 1