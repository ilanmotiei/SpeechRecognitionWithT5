
import torch
import numpy as np

training_data_root = 'LibriSpeech/train-clean-100+360+500_ordered'
validation_data_root = 'LibriSpeech/dev-clean_ordered'
sample_rate = 16000

batch_size = 32
lr = 1e-03
num_beams = 4  # for beam search

n_mels = 512
n_fft = 400
power = np.e
window_length = 400
hop_len = 160
augment_rate = 0.5  # half of the training data will be augmented with the following 3 parameters:
augment_frequency_mask_param = 80  # 80 out of 512 mel-bin rows will be masked
augment_time_mask_rate = 0.05  # 5% of time steps (columns) will be masked
augment_time_mask_amount = 1

max_audio_duration = 5  # in seconds

embedding_dim = 512

device = torch.device('cuda')
n_gpus = 1
epochs = 1000
num_workers = 4 * n_gpus

