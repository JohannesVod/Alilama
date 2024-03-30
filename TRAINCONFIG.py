class ModelParams:
    model_name:str = "alilama"
    d_model:int = 128
    blocks:int = 8
    max_seq_len:int = 128
    num_heads:int = 8
    hidden_dim:int = 4*d_model
    head_width:int = d_model//num_heads

# change model Parameters here:
model_name = "alilama"
d_model = 512 # embedding dimension
blocks = 4 # number of transformer blocks that are repeated
max_seq_len = 128 # sequence length the transformer is trained on
num_heads = 8 # heads in the multiheaded attention
hidden_dim = 4*d_model # size of linear transformation inside MLP
head_width = d_model//num_heads # width of head
assert head_width * num_heads == d_model
# experimental! I'm trying to reduce model parameters by weight tying certain
# transformer blocks with each other. This is a good pattern that i discovered:
weight_tying_dict = {}
# {0:3, 1:4, 2:5, 3:6, 4:7}: ~1.31 at 400000
# {2:3, 1:5, 2:6, 3:7} ~1.33 at 400000 with 1496064 params
# {3:1, 5:1, 4:2, 6:2} ~1.32 at 400000 with 1496064 params
# Training Parameters
BATCH_SIZE = 248
learning_rate = 5e-5 # set to 1e-4 at start, then at 1e-5
train_steps = 1000000 # Numbers of batches used for training
beta1 = 0.9 # Parameter for adam optimizer
beta2 = 0.95
weight_decay = 0 # used to reduce overfitting
compile_model = False # compiles the model but takes some time

# Logging:
log_interval = 5 # interval where we print the loss function
eval_interval = log_interval*20 # interval where we evaluate the model
eval_batches = 1