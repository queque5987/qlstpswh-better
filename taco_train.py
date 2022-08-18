from train import train
from taco_inference import taco_config

if __name__ == "__main__":
    hparams = taco_config()
    hparams.training_files='C:/test/models/filelist.txt'
    hparams.validation_files='C:/test/models/filelist.txt'
    train("C:/test/models",
        "C:/test/models",
        "C:/test/models/checkpoint_9000",
        True, 1, 0, "group_name", hparams)
