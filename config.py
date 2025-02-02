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
    img_paths = "./data/imgs.pkl"
    feature_save_path = "./data/features.pkl"
    label_save_path = "./data/labels.pkl"
    model_dir = "./checkpoints/"
    restore_index = 4851

    # Hyperparameter
    max_epoch = 500
    batch_size = 16
    embedding_size = 512
    lr = 0.001
    moving_average_decay = 0.999
    weight_decay = 5e-4
    keep_prob = 0.8

    # Data
    data_split_ratio = 0.8
    threads = 4

    # For center loss
    alpha = 0.6
    center_loss_factor = 1e-2
    ######## For PILLNET training ########

    ######## For PILLNET recognition ########
    max_boxes_to_draw = 20
    min_score_thresh = .5
    embed_threshold = 0.02
    ######## For PILLNET recognition ########


opt = Config()