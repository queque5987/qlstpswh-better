import argparse

import text
from text import symbols
from model import Tacotron2

import torch
import numpy as np
import matplotlib.pylab as plt
import numpy as np

class taco_config():
  def __init__(self):
    self.epochs=20
    self.iters_per_checkpoint=1000
    self.seed=1234
    self.dynamic_loss_scaling=True
    self.fp16_run=False
    self.distributed_run=False
    self.dist_backend="nccl"
    self.dist_url="tcp://localhost:54321"
    self.cudnn_enabled=True
    self.cudnn_benchmark=False
    self.ignore_layers=['embedding.weight']

    ################################
    # Data Parameters             #
    ################################
    self.load_mel_from_disk=False
    self.training_files='/content/drive/MyDrive/Vinsenjo/tacotaco/train-clean-360_filelist.txt'
    self.validation_files='/content/drive/MyDrive/Vinsenjo/tacotaco/dev-clean_filelist.txt'
    self.text_cleaners=['english_cleaners']

    ################################
    # Audio Parameters             #
    ################################
    self.max_wav_value=32768.0
    self.sampling_rate=16000 #22050
    self.filter_length=1024
    self.hop_length=256
    self.win_length=1024
    self.n_mel_channels=80
    self.mel_fmin=0.0
    self.mel_fmax=8000.0

    ################################
    # Model Parameters             #
    ################################
    self.n_symbols=len(symbols)
    self.symbols_embedding_dim=512

    # Encoder parameters
    self.encoder_kernel_size=5
    self.encoder_n_convolutions=3
    self.encoder_embedding_dim=512

    # Decoder parameters
    self.n_frames_per_step=1  # currently only 1 is supported
    self.decoder_rnn_dim=1024
    self.prenet_dim=256
    self.max_decoder_steps=2000 #1000
    self.gate_threshold=0.5
    self.p_attention_dropout=0.1
    self.p_decoder_dropout=0.1

    # Attention parameters
    self.attention_rnn_dim=1024
    self.attention_dim=128

    # Location Layer parameters
    self.attention_location_n_filters=32
    self.attention_location_kernel_size=31

    # Mel-post processing network parameters
    self.postnet_embedding_dim=512
    self.postnet_kernel_size=5
    self.postnet_n_convolutions=5

    ################################
    # Optimization Hyperparameters #
    ################################
    self.use_saved_learning_rate=False
    self.learning_rate=1e-3
    self.weight_decay=1e-6
    self.grad_clip_thresh=1.0
    self.batch_size=12
    self.mask_padding=True  # set model's padded outputs to padded values

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='upper', 
                       interpolation='none')
    plt.show()

def get_taco(ckpt_to_use = "C:/test/models/checkpoint_5.pt"):
    hparams = taco_config()
    model = Tacotron2(hparams)
    checkpoint_path = ckpt_to_use
    checkpoint_dict = torch.load(
        checkpoint_path,
        map_location='cpu')
    # model.load_state_dict(checkpoint_dict['state_dict'])
    model.load_state_dict(checkpoint_dict)
    model.eval()
    return model, hparams

def get_mel_outputs(texts = "i didn't preach with out direction"):
    model, hparams = get_taco()
    sequence = np.array(text.text_to_sequence(texts, hparams.text_cleaners))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    plot_data((mel_outputs.float().data.cpu().numpy()[0],
            mel_outputs_postnet.float().data.cpu().numpy()[0],
            alignments.float().data.cpu().numpy()[0].T))
    return mel_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('texts')
    args = parser.parse_args()
    texts = args.texts
    get_mel_outputs(texts)
    