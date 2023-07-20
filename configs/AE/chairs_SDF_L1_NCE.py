{
    "Description" : [ "This experiment learns a shape representation for chairs ",
                      "using data from ShapeNet version 1." ],
    "DataSource" : "/scratch/",
    "TrainSplit" : "splits/chairs_yue_train.json",
    "TestSplit" : "splits/chairs_yue_test.json",
    "AugmentData" : false,
    "NumberOfViews" : 16,
    "NetworkEncoder" : "encoder_pointnet",
    "NormType": "in",
    "Depth": 20,
    "NetworkDecoder" : "decoder",
    "NetworkSpecs" : {
      "dims" : [ 512, 512, 512, 512, 512],
      "dropout" : [0, 1, 2, 3],
      "dropout_prob" : 0.0,
      "norm_layers" : [0, 1, 2, 3],
      "xyz_in_all" : false,
      "use_tanh" : false,
      "latent_dropout" : false,
      "weight_norm" : true
      },
    "CodeLength" : 256,
    "NumEpochs" : 2001,
    "SnapshotFrequency" : 1000,
    "AdditionalSnapshots" : [],
    "LearningRateSchedule" : [
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
      }],
    "SamplesPerScene" : 8192,
    "ScenesPerBatch" : 16,
    "DataLoaderThreads" : 12,
    "ClampingDistance" : 0.1,
    "CodeRegularization" : true,
    "CodeRegularizationLambda" : 1e-4,
    "CodeBound" : 1.0
  }