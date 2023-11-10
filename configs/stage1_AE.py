# only train encoder for sketch, fix shape latent code 
train_shape_encoder = True

# Encoder
CodeLength = 256
load_AE_model = True
encoder_epoch = 100
without_sigmoid = True
num_points = 4096

### setting for SDF decoder
NetworkSpecs = {
    "dims" :  [ 512, 512, 512, 512, 512],
    "dropout" :  [0, 1, 2, 3],
    "dropout_prob" :  0.0,
    "norm_layers" : [0, 1, 2, 3],
    "latent_in" : [],
    "xyz_in_all" : False,
    "use_tanh" : False,
    "latent_dropout" : False,
    "weight_norm" : True
    }

# Training loss handles
train_with_emb_loss = ['L1', 'NCE', 'sketch_sdf']
train_with_nf_loss = False
train_with_sketch_sdf_loss = False
train_with_latent_loss = False
train_with_sigma_norm_loss = False

# Triplet loss
margin = 0.3

eval_epoch_fun = 'eval_one_epoch_AE'
