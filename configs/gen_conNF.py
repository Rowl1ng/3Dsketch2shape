# only train encoder for sketch, fix shape latent code 
train_shape_encoder = False

### setting for point cloud encoder
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


### setting for Autoencoder
TrainSplit = 'splits/chairs_yue_train.json'
TestSplit = 'splits/chairs_yue_test.json'

NumEpochs = 2001
SnapshotFrequency = 1000
AdditionalSnapshots = []
LearningRateSchedule = [
      {
        "Type" : "Step",
        "Initial" : 0.0005,
        "Interval" : 500,
        "Factor" : 0.5
      },
      {
        "Type" : "Step",
        "Initial" : 0.001,
        "Interval" : 500,
        "Factor" : 0.5
      }]


### setting for Flow model

# Training loss handles
train_with_emb_loss = []
train_with_nf_loss = True
train_with_sketch_sdf_loss = True
train_with_latent_loss = False
train_with_sigma_norm_loss = False

lambda_sketch_sdf_loss = 100


resume_ckpt = 'recon_L1+NCE+sketch_SDF_run1/model_epoch_300.pth'

# Conditional NF
conditional_NF = True
eval_epoch_fun = 'eval_one_epoch_v4'

freeze_encoder = True
num_samples = 8
