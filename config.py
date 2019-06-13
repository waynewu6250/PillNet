class Config:

    ########## For LiveStream ############
    MODEL_NAME = './align/my_exported_graphs'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = './align/training/pill_detection.pbtxt'
    NUM_CLASSES = 1
    image_path = './align/raw_data/tests/YuLuAn_Cold_FC_Tablets.jpg'
    ########## For LiveStream ############

    ######## For PILLNET training ########
    data_path = "./data/train_imgs"
    model_dir = "./checkpoints/"

    # Hyperparameter
    max_epoch = 100
    batch_size = 16
    embedding_size = 512
    lr = 0.01
    moving_average_decay = 0.999
    weight_decay = 5e-4
    keep_prob = 0.8

    # Data
    img_size = (160, 160)
    data_split_ratio = 0.8
    threads = 4

    # For center loss
    alpha = 0.6
    center_loss_factor = 1e-2
    ######## For PILLNET training ########


opt = Config()