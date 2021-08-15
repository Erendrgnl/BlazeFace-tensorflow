

def get_hyper_parameters():
    c= {}
    c["input_shape"] = [128, 128, 3]
    c["batch_size"] = 32
    c["epoch"] = 150
    c["detections_per_layer"] = [2, 6]
    c["train"] = True
    c["num_data"] = 48980
    c["checkpoint_path"] = "./saved/"
    c["dataset_dir"] = None
    c["label_path"] = None

    return c