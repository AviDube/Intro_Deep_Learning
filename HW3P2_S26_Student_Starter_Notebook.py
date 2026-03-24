import os
import pprint
import torch
import random
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import wandb
import torchaudio.transforms as tat
from torchaudio.models.decoder import cuda_ctc_decoder
import Levenshtein

from sklearn.metrics import accuracy_score
import gc

import glob

import zipfile
from tqdm.auto import tqdm
import os
import datetime


import warnings
warnings.filterwarnings('ignore')

import faulthandler
import sys
import os
import kaggle

#TODO: Only set Acknowledged to True if you have read the above acknowledgements and agree to ALL of them.
ACKNOWLEDGED = True # @param {"type":"boolean","placeholder":"False"}

os.environ['KAGGLE_USERNAME'] = "avidube" # TODO: Verify in Settings
os.environ['KAGGLE_API_TOKEN'] = "KGAT_e4a59dacaeaefc1cf225368be254aeae" # TODO: Add Access Token (must be the new token starting with "KGAT_")
api = kaggle.api  # Already authenticated on import
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)


# # Config

# In[49]:
import yaml
with open("config.yaml") as file:
    config = yaml.safe_load(file)

BATCH_SIZE = config["batch_size"] # Define batch size from config
root = "/home/avid/Intro_Deep_Learning/11785-hw3p2" # TODO: Specify the directory to your root based on your environment


# # Setup Vocabulary (DO NOT MODIFY)

# In[47]:


# ARPABET PHONEME MAPPING
# DO NOT CHANGE

CMUdict_ARPAbet = {
    "" : " ",
    "[SIL]": "-", "NG": "G", "F" : "f", "M" : "m", "AE": "@",
    "R"    : "r", "UW": "u", "N" : "n", "IY": "i", "AW": "W",
    "V"    : "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R",
    "HH"   : "h", "Z" :
     "z", "K" : "k", "CH": "C", "W" : "w",
    "EY"   : "e", "ZH": "Z", "T" : "t", "EH": "E", "Y" : "y",
    "AH"   : "A", "B" : "b", "P" : "p", "TH": "T", "DH": "D",
    "AO"   : "c", "G" : "g", "L" : "l", "JH": "j", "OY": "O",
    "SH"   : "S", "D" : "d", "AY": "Y", "S" : "s", "IH": "I",
    "[SOS]": "[SOS]", "[EOS]": "[EOS]"
}


CMUdict = list(CMUdict_ARPAbet.keys())
ARPAbet = list(CMUdict_ARPAbet.values())

PHONEMES = CMUdict[:-2] #To be used for mapping original transcripts to integer indices
LABELS = ARPAbet[:-2] #To be used for mapping predictions to strings

OUT_SIZE = len(PHONEMES) # Number of output classes
print("Number of Phonemes:", OUT_SIZE)

# Indexes of BLANK and SIL phonemes
BLANK_IDX=CMUdict.index('')
SIL_IDX=CMUdict.index('[SIL]')

print("Index of Blank:", BLANK_IDX)
print("Index of [SIL]:", SIL_IDX)

# # Dataset and Dataloader

# ### Train Data

# In[56]:


class AudioDataset(torch.utils.data.Dataset):

    # For this homework, we give you full flexibility to design your data set class.
    # Hint: The data from HW1 is very similar to this HW

    #TODO
    def __init__(self, root, apply_transformations=False, partition="train-clean-100"):
        '''
        Initializes the dataset.

        INPUTS: What inputs do you need here?
        '''

        self.PHONEMES = PHONEMES
        self.subset = config['subset']

        # TODO
        # Define the directories containing MFCC and transcript files
        self.mfcc_dir = os.path.join(root, partition, "mfcc") #TODO
        self.transcript_dir = os.path.join(root, partition, "transcript") #TODO

        # List all files in the directories. Remember to sort the files
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir)) #TODO
        print(f"Number of MFCC files found: {len(self.mfcc_files)}")
        self.transcript_files = sorted(os.listdir(self.transcript_dir)) #TODO

        # Compute size of data subset
        subset_size = int(self.subset * len(self.mfcc_files))

        # Select subset of data to use
        self.mfcc_files = self.mfcc_files[:subset_size]
        self.transcript_files = self.transcript_files[:subset_size]

        assert(len(self.mfcc_files) == len(self.transcript_files))

        #TODO
        # CREATE AN ARRAY TO STORE ALL PROCESSED MFCCS AND TRANSCRIPTS
        # LOAD ALL MFCCS AND CORRESPONDING TRANSCRIPTS AND DO THE NECESSARY PRE-PROCESSING
          # HINTS:
          # WHAT NORMALIZATION TECHNIQUE DID YOU USE IN HW1? CAN WE USE IT HERE?
          # REMEMBER TO REMOVE [SOS] AND [EOS] FROM TRANSCRIPTS
        '''
        You may decide to do this in __getitem__ if you wish.
        However, doing this here will make the __init__ function take the load of
        loading the data, and shift it away from training.
        '''
        self.mfccs, self.transcripts = [], []
        for i in tqdm(range(len(self.mfcc_files))):

            # TODO: Load a single mfcc. Hint: Use numpy
            mfcc             = np.load(os.path.join(self.mfcc_dir, self.mfcc_files[i]))
            # TODO: Do Cepstral Normalization of mfcc along the Time Dimension (Think about the correct axis)
            mfccs_normalized = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0)) #+ 1e-8)

            # Convert mfcc to tensor
            mfccs_normalized = torch.tensor(mfccs_normalized, dtype=torch.float32)

            # TODO: Load the corresponding transcript
            # Remove [SOS] and [EOS] from the transcript
            # (Is there an efficient way to do this without traversing through the transcript?)
            # Note that SOS will always be in the starting and EOS at end, as the name suggests.
            transcript = np.load(os.path.join(self.transcript_dir, self.transcript_files[i]))
            transcript = transcript[1:-1]

            # The available phonemes in the transcript are of string data type
            # But the neural network cannot predict strings as such.
            # Hence, we map these phonemes to integers

            #TODO
            # HOW CAN WE REPRESENT PHONEMES? CAN WE CREATE A MAPPING FOR THEM?
            # HINT: TENSORS CANNOT STORE NON-NUMERICAL VALUES OR STRINGS
            # TODO: Map the phonemes to their corresponding list indexes in self.phonemes
            transcript_indices = np.array([self.PHONEMES.index(p) for p in transcript])
            # Now, if an element in the transcript is 0, it means that it is 'SIL' (as per the above example)

            # Convert transcript to tensor
            transcript_indices = torch.tensor(transcript_indices, dtype=torch.int64)

            # Append each mfcc to self.mfcc, transcript to self.transcript
            self.mfccs.append(mfccs_normalized)
            self.transcripts.append(transcript_indices)

        # self.mfccs          = torch.concatenate(self.mfccs, dim=0)
        # self.transcripts    = torch.concatenate(self.transcripts, dim=0)

        #TODO
        # WHAT SHOULD THE LENGTH OF THE DATASET BE?
        self.length = len(self.mfccs)
 #TODO

    def __len__(self):

        '''
        TODO: What do we return here?
        '''

        return self.length #TODO


    def __getitem__(self, ind):

        '''
        TODO: RETURN THE MFCC COEFFICIENTS AND ITS CORRESPONDING LABELS

        If you didn't do the loading and processing of the data in __init__,
        do that here.

        Once done, return a tuple of features and labels.

        '''

        mfcc = self.mfccs[ind] #TODO
        transcript = self.transcripts[ind] #TODO

        # NOTE: Remember to convert mfcc and transcripts to tensors here, if not done already in __init__
        return mfcc, transcript


    def collate_fn(self,batch):
        '''
        TODO:
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish.
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features,
            and lengths of labels.
        '''

        # Extract batch of input MFCCs and batch of output transcripts separately
        batch_mfcc = [item[0] for item in batch] #TODO
        batch_transcript = [item[1] for item in batch] #TODO

        # Store original lengths of the MFCCS and transcripts in the batches
        lengths_mfcc = [len(mfcc) for mfcc in batch_mfcc] #TODO
        lengths_transcript = [len(transcript) for transcript in batch_transcript] #TODO

        # Pad the MFCC sequences and transcripts
        # HINT: CHECK OUT -> pad_sequence (imported above)
        # Also be sure to check the input format (batch_first)
        # Note: (resulting shape of padded MFCCs: [batch, time, freq])
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True) #TODO
        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True) #TODO


        # TODO: You may apply some transformations, Time and Frequency masking, here in the collate function;
        # Food for thought -> Why are we applying the transformation here and not in the __getitem__?
        #                  -> Would we apply transformation on the validation set as well?
        #                  -> Is the order of axes / dimensions as expected for the transform functions?
        #                     -> Time & Freq. Masking functions both expect input of shape (..., freq, time),
        #                        So permute your input dimensions appropriately before & after using these functions.



        # Return the following values: padded features, padded labels, actual length of features, actual length of the labels
        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)


# ### Test Data

# In[53]:


# TODO
# Food for thought -> Do you need to apply transformations in this test dataset class?
class AudioDatasetTest(torch.utils.data.Dataset):

    def __init__(self, root, partition="test-clean"):
        self.PHONEMES = PHONEMES
        self.mfcc_dir = os.path.join(root, partition, "mfcc")
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir)) #TODO
        self.mfccs = []
        for i in tqdm(range(len(self.mfcc_files))):
            mfcc = np.load(os.path.join(self.mfcc_dir, self.mfcc_files[i]))
            mfccs_normalized = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
            mfccs_normalized = torch.tensor(mfccs_normalized, dtype=torch.float32)
            self.mfccs.append(mfccs_normalized)

        # self.mfccs = torch.concatenate(self.mfccs, dim=0)
        self.length = len(self.mfccs)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        mfcc = self.mfccs[ind]
        return mfcc

    def collate_fn(self, batch):
        batch_mfcc = batch
        lengths_mfcc = [len(mfcc) for mfcc in batch_mfcc]
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)

        return batch_mfcc_pad, torch.tensor(lengths_mfcc)


# ### Data loaders

# In[57]:


# To free up ram
import gc
gc.collect()


# In[58]:


# Create objects for the dataset classes
train_data = AudioDataset(root, apply_transformations=True, partition="train-clean-100") # TODO: Fill in the required parameters
val_data = AudioDataset(root, apply_transformations=False, partition="dev-clean") # TODO: You can either use the same class for train data with some modifications or make a new one :)
test_data = AudioDatasetTest(root, partition="test-clean") # TODO: Fill in the required parameters


# In[14]:


# Do NOT forget to pass in the collate function as an argument while creating the dataloader
train_loader = torch.utils.data.DataLoader(
    dataset     = train_data,
    num_workers = 4,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = True,
    collate_fn = train_data.collate_fn
)

val_loader = torch.utils.data.DataLoader(
    dataset     = val_data,
    num_workers = 0,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = False,
    collate_fn = val_data.collate_fn
)

test_loader = torch.utils.data.DataLoader(
    dataset     = test_data,
    num_workers = 0,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = False,
    collate_fn = test_data.collate_fn
)

print("Batch size: ", config['batch_size'])
print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

# sanity check
for data in train_loader:
    x, y, lx, ly = data
    print(x.shape, y.shape, lx.shape, ly.shape)
    break


# # Network

# ## Basic network (Optional)
# 
# This is a basic block for understanding, you can skip this and move to pBLSTM one

# In[17]:


torch.cuda.empty_cache()

class Network(nn.Module):

    def __init__(self):

        super(Network, self).__init__()

        # TODO: Adding some sort of embedding layer or feature extractor might help performance.
        # You can use CNNs as Embedding layer to extract features. Keep in mind the Input dimensions and expected dimension of Pytorch CNN.
        # Food for thought -> What type of Conv layers can be used here?
        #                  -> What should be the size of input channels to the first layer?
        self.embedding = nn.Conv1d(28, 256, kernel_size=1) #TODO

        # TODO : look up the documentation. You might need to pass some additional parameters.
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=3, batch_first=True) #TODO

        self.classification = nn.Sequential(
            nn.Linear(256, OUT_SIZE)  #TODO: Linear layer with in_features from the lstm module above and out_features = OUT_SIZE
        )


        self.logSoftmax = nn.LogSoftmax(dim=2) #TODO: Apply a log softmax here. Which dimension would apply it on ?

    def forward(self, x, lx):
        #TODO
        # The forward function takes 2 parameter inputs here. Why?
        # Refer to the handout for hints
        x_em = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        pack_seq = pack_padded_sequence(x_em, lx, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(pack_seq)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.classification(x)
        x = self.logSoftmax(x)
        return x


# ## Initialize Basic Network
# (If trying out the basic Network)

# In[18]:


torch.cuda.empty_cache()

model = Network().to(device)

# Check to stay below 20 MIL Parameter limit
assert sum(p.numel() for p in model.parameters() if p.requires_grad) < 20_000_000, "Exceeds 20 MIL params. Any submission made to Kaggle with this model will be flagged as an AIV."

print(model)


# 
# ## ASR Network
# We define everything we need for the ASR model in separate classes, and put them all together in the end

# #### Permute class

# In[19]:


class Permute(torch.nn.Module):
    '''
    Used to transpose/permute the dimensions of an MFCC tensor.
    '''
    def forward(self, x):
        return x.transpose(1, 2)


# #### Pyramidal Bi-LSTM (pBLSTM) class

# In[20]:


class pBLSTM(torch.nn.Module):

    '''
    Pyramidal BiLSTM
    Read the write up/paper and understand the concepts and then write your implementation here.

    At each step,
    1. Pad your input if it is packed (Unpack it)
    2. Reduce the input length dimension by concatenating feature dimension
        (Tip: Write down the shapes and understand)
        (i) How should  you deal with odd/even length input?
        (ii) How should you deal with input length array (x_lens) after truncating the input?
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    '''

    def __init__(self, input_size, hidden_size):
        super(pBLSTM, self).__init__()

        self.blstm = nn.LSTM(input_size * 2, hidden_size, bidirectional=True) # TODO: Initialize a single layer bidirectional LSTM with the given input_size and hidden_size

    def forward(self, x_packed): # x_packed is a PackedSequence
        # TODO: Pad Packed Sequence
        x, x_lens = pad_packed_sequence(x_packed, batch_first=True)

        # TODO: Call self.trunc_reshape() which downsamples the time steps of x and increases the feature dimensions as mentioned above
        # self.trunc_reshape will return 2 outputs. What are they? Think about what quantites are changing.
        x, x_lens = self.trunc_reshape(x, x_lens)

        # TODO: Pack Padded Sequence. What output(s) would you get?
        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        # TODO: Pass the sequence through bLSTM
        output, _ = self.blstm(x_packed)

        # What do you return?
        return output #TODO

    def trunc_reshape(self, x, x_lens):

        # TODO: If you have odd number of timesteps, how can you handle it? (Hint: You can exclude them)
        batch_size, max_len, features = x.shape
        if max_len % 2 == 1:
            x = x[:, :-1, :]

        # TODO: Reshape x. When reshaping x, you have to reduce number of timesteps by a downsampling factor while increasing number of features by the same factor
        x = x.reshape(batch_size, -1, features * 2)

        # TODO: Reduce lengths by the same downsampling factor
        x_lens = x_lens // 2

        return x, x_lens


# #### Util for LSTM
# 

# In[21]:


class LSTMWrapper(torch.nn.Module):
    '''
    Used to get only output of lstm, not the hidden states.
    '''
    def __init__(self, lstm):
        super(LSTMWrapper, self).__init__()
        self.lstm = lstm

    def forward(self, x):
        output, _ = self.lstm(x)
        return output


# #### Encoder class

# In[22]:


class Encoder(torch.nn.Module):
    '''
    The Encoder takes utterances as inputs and returns latent feature representations
    '''
    def __init__(self, input_size, encoder_hidden_size):
        super(Encoder, self).__init__()


        # TODO: You can use CNNs as Embedding layer to extract features. Keep in mind the Input dimensions and expected dimension of Pytorch CNN.
        # Food for thought -> What type of Conv layers can be used here?
        #                  -> What should be the size of input channels to the first layer?
        self.embedding = nn.Conv1d(input_size, 128, kernel_size=1) #TODO

        # TODO:
        self.BLSTMs = LSTMWrapper(
            # TODO: Look up the documentation. You might need to pass some additional parameters.
            torch.nn.LSTM(input_size=128, hidden_size=encoder_hidden_size, num_layers=2, bidirectional=True, batch_first=True) #TODO
          )

        self.pBLSTMs = torch.nn.Sequential( # How many pBLSTMs are required?
            # TODO: Fill this up with pBLSTMs - What should the input_size be?
            # Hint: You are downsampling timesteps by a factor of 2, upsampling features by a factor of 2 and the LSTM is bidirectional)
            # Optional: Dropout/Locked Dropout after each pBLSTM (Not needed for early submission)
            # https://github.com/salesforce/awd-lstm-lm/blob/dfd3cb0235d2caf2847a4d53e1cbd495b781b5d2/locked_dropout.py#L5
            pBLSTM(2 * encoder_hidden_size, encoder_hidden_size)
        )

    def forward(self, x, x_lens):
        # Where are x and x_lens coming from? The dataloader

        # TODO: Call the embedding layer
        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)

        # TODO: Pack Padded Sequence
        # Note: pack_padded_sequence requires lengths on CPU
        x_packed = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)

        # TODO: Pass Sequence through the Bi-LSTM layer
        x_packed = self.BLSTMs(x_packed)

        # TODO: Pass Sequence through the pyramidal Bi-LSTM layer
        for pblstm in self.pBLSTMs:
            x_packed = pblstm(x_packed)
            x_lens = x_lens // 2

        # TODO: Pad Packed Sequence
        encoder_outputs, encoder_lens = pad_packed_sequence(x_packed, batch_first=True)

        # Remember the number of output(s) each function returns
        return encoder_outputs, encoder_lens


# #### Decoder class

# In[23]:


class Decoder(torch.nn.Module):

    def __init__(self, embed_size, output_size= 41):
        super().__init__()

        self.mlp = torch.nn.Sequential(

            Permute(),
            torch.nn.BatchNorm1d(2 * embed_size),
            Permute(),

            #TODO define your MLP arch. Refer HW1P2
            #Use Permute Block before and after BatchNorm1d() to match the size
            #Now you can stack your MLP layers
            torch.nn.Linear(2 * embed_size, embed_size),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_size, output_size)
        )

        self.softmax = torch.nn.LogSoftmax(dim=2)


    def forward(self, encoder_out):
        out = self.mlp(encoder_out)

        #TODO: Think about what should be the final output of the decoder for classification
        out = self.softmax(out)

        return out


# #### ASR Model Class

# In[24]:


class ASRModel(torch.nn.Module):

    def __init__(self, input_size, embed_size= 192, output_size= len(PHONEMES)):
        super().__init__()

        # Initialize encoder and decoder
        self.encoder        = Encoder(input_size, embed_size) # TODO: Initialize Encoder
        self.decoder        = Decoder(embed_size, output_size) # TODO: Initialize Decoder


    def forward(self, x, lengths_x):

        encoder_out, encoder_lens   = self.encoder(x, lengths_x)
        decoder_out                 = self.decoder(encoder_out)

        return decoder_out, encoder_lens


# ## Initialize ASR Network

# In[25]:


model = ASRModel(
    input_size  = 28,  #TODO
    embed_size  = config['embed_size'], #TODO
    output_size = len(PHONEMES)
).to(device)

# Check to stay below 20 MIL Parameter limit
assert sum(p.numel() for p in model.parameters() if p.requires_grad) < 20_000_000, "Exceeds 20 MIL params. Any submission made to Kaggle with this model will be flagged as an AIV."


# In[26]:


summary(model, input_data=[x.to(device), lx.to(device)])


# # Training Config
# Initialize Loss Criterion, Optimizer, CTC Beam Decoder, Scheduler, Scaler (Mixed-Precision), etc

# In[27]:


# TODO: Define CTC loss as the criterion. How would the losses be reduced?
criterion = nn.CTCLoss(reduction='mean', blank=BLANK_IDX) #TODO
# CTC Loss: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
# Refer to the handout for hints

optimizer =  torch.optim.AdamW(model.parameters(), lr=config['learning_rate']) #TODO: What goes in here?

# TODO: Declare the decoder. Use the PyTorch Cuda CTC Decoder to decode phonemes
# CTC Decoder: https://pytorch.org/audio/2.1/generated/torchaudio.models.decoder.cuda_ctc_decoder.html
decoder = cuda_ctc_decoder(tokens=LABELS, nbest=1, beam_size=config['train_beam_width']) #TODO

# TODO:
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #TODO

# Mixed Precision, if you need it
scaler = torch.cuda.amp.GradScaler()


# ### Levenshtein

# In[28]:


def decode_prediction(output, output_lens, decoder, PHONEME_MAP = LABELS):

    # Look at docs for CUDA_CTC_DECODER for more info on how it was used here:
    # https://pytorch.org/audio/main/tutorials/asr_inference_with_cuda_ctc_decoder_tutorial.html
    print("Started the decoding process")
    output = output.contiguous()
    print("Output is contiguous")
    output_lens = output_lens.to(torch.int32).contiguous()
    print("Output lengths are contiguous")
    beam_results = decoder(output, output_lens.to(torch.int32)) #lengths - list of lengths
    print("Beam created")
    pred_strings = []

    for i in range(len(beam_results)):
        # Get the top prediction for each sequence in the batch
        top_beam_results = beam_results[i][0].tokens

        # Map the sequence of indices to actual phoneme LABELS and join them into a string
        pred_string = "".join([PHONEME_MAP[token] for token in top_beam_results])

        # Append to predicted strings list
        pred_strings.append(pred_string)

    return pred_strings

def calculate_levenshtein(output, label, output_lens, label_lens, decoder, PHONEME_MAP= LABELS): # y - sequence of integers

    dist            = 0
    batch_size      = label.shape[0]

    pred_strings    = decode_prediction(output, output_lens, decoder, PHONEME_MAP)

    for i in range(batch_size):
        # TODO: Get predicted string and label string for each element in the batch
        label_string = ''.join([PHONEME_MAP[idx.item()] for idx in label[i, :label_lens[i]]])
        pred_string = pred_strings[i]  # TODO: Predicted string from decode_prediction

        dist += Levenshtein.distance(pred_string, label_string)

    # Average the distance over the batch
    dist /= batch_size # Think about why we are doing this
    return dist


# In[29]:


torch.cuda.empty_cache()
gc.collect()


# ## Test Implementation

# In[30]:


# test code to check shapes

model.eval()
for i, data in enumerate(val_loader, 0):
    x, y, lx, ly = data
    x, y = x.to(device), y.to(device)
    lx, ly = lx.to(device), ly.to(device)
    h, lh = model(x, lx)
    print(h.shape)
    h = torch.permute(h, (1, 0, 2))
    print(h.shape, y.shape)
    loss = criterion(h, y, lh, ly)
    print(loss)

    print(calculate_levenshtein(torch.permute(h, (1, 0, 2)), y, lh.to(device), ly, decoder, LABELS))

    del x, y, lx, ly, h, lh, loss
    torch.cuda.empty_cache()

    break


# ## WandB

# In[31]:


# Use wandb? Resume Training?
USE_WANDB = config['wandb']

RESUME_LOGGING = False # Set this to true if you are resuming training from a previous run

# Create your wandb run

run_name = '{}_early_submission'.format(config['Name'])

# If you are resuming an old run
if USE_WANDB:

    wandb.login(key="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") #TODO

    if RESUME_LOGGING:
        run = wandb.init(
            id     = "", ### Insert specific run id here if you want to resume a previous run
            resume = "must", ### You need this to resume previous runs
            project = "hw3p2", ### Project should be created in your wandb
            settings = wandb.Settings(_service_wait=300)
        )


    else:
        run = wandb.init(
            name    = run_name, ### Wandb creates random run names if you skip this field, we recommend you give useful names
            reinit  = True, ### Allows reinitalizing runs when you re-run this cell
            project = "hw3p2", ### Project should be created in your wandb account
            config  = config ### Wandb Config for your run
        )

        ### Save your model architecture as a string with str(model)
        model_arch  = str(model)
        ### Save it in a txt file
        arch_file   = open("model_arch.txt", "w")
        file_write  = arch_file.write(model_arch)
        arch_file.close()

        ### log it in your wandb run with wandb.save()
        wandb.save('model_arch.txt')


# # Training Functions

# In[32]:


# Train function
def train_model(model, train_loader, criterion, optimizer):

    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    total_loss = 0

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)
        lx, ly = lx.to(device), ly.to(device)

        with torch.cuda.amp.autocast():
            h, lh = model(x, lx)
            h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += loss.item()

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update() # Update tqdm bar

        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() # This is something added just for FP16

        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()

    batch_bar.close() # You need this to close the tqdm bar

    return total_loss / len(train_loader)


# Eval function
def validate_model(model, val_loader, decoder, phoneme_map= LABELS):

    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    total_loss = 0
    vdist = 0

    for i, data in enumerate(val_loader):

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)
        lx, ly = lx.to(device), ly.to(device)

        with torch.inference_mode():
            h, lh = model(x, lx)
            h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += loss.item()
        vdist += calculate_levenshtein(torch.permute(h, (1, 0, 2)), y, lh.to(device), ly, decoder, phoneme_map)

        batch_bar.set_postfix(loss="{:.04f}".format(float(total_loss / (i + 1))), dist="{:.04f}".format(float(vdist / (i + 1))))

        batch_bar.update()

        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()

    batch_bar.close()
    total_loss = total_loss/len(val_loader)
    val_dist = vdist/len(val_loader)
    return total_loss, val_dist


# ### Model Saving & Loading functions

# In[33]:


def save_model(model, optimizer, scheduler, metric, epoch, path):
    torch.save(
        {'model_state_dict'         : model.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict(),
         'scheduler_state_dict'     : scheduler.state_dict() if scheduler is not None else {},
         metric[0]                  : metric[1],
         'epoch'                    : epoch},
         path
    )

def load_model(path, model, optimizer= None, scheduler= None, metric='valid_dist'):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler != None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch   = checkpoint['epoch']
    metric  = checkpoint[metric]

    print("\nResuming training from epoch:", epoch)
    print('----------------------------------------\n')
    print("Epochs left: ", config['epochs'] - epoch)
    print("Optimizer: \n", optimizer)
    # print("Current Schedueler T_cur:", scheduler.T_cur)

    print("Best Val Dist:", metric)

    return [model, optimizer, scheduler, epoch, metric]


# ## Training Loop

# In[34]:


# Instantiate variables used in training loop
last_epoch_completed = 0
best_lev_dist = float("inf")


# #### Uncomment this if resuming training from model checkpoint

# In[35]:


# RESUME_TRAINING = True # Set this to true if you are resuming training from a model checkpoint

# if RESUME_TRAINING:

#     checkpoint_path = ''
#     checkpoint = load_model(checkpoint_path, model, optimizer, scheduler, metric='valid_dist')

#     last_epoch_completed = checkpoint[3]
#     best_lev_dist = checkpoint[4]


# In[36]:


# Set up checkpoint directories and WanDB logging watch
checkpoint_root = os.path.join(os.getcwd(), 'checkpoints')
os.makedirs(checkpoint_root, exist_ok=True)
wandb.watch(model, log="all")

checkpoint_best_model_filename = 'checkpoint-best-model.pth'
checkpoint_last_epoch_filename = 'checkpoint-last-epoch.pth'
epoch_model_path = os.path.join(checkpoint_root, checkpoint_last_epoch_filename)
best_model_path = os.path.join(checkpoint_root, checkpoint_best_model_filename)

# WanDB log watch
if config['wandb']:
  wandb.watch(model, log="all")


# In[37]:


# Clear RAM for storage before you start training
torch.cuda.empty_cache()
gc.collect()


# #### Iterate over the number of epochs to train and evaluate your model
# 

# In[38]:


# Debug: Check sequence lengths before training
print("=== DEBUGGING SEQUENCE LENGTHS ===")
x_sample, y_sample, lx_sample, ly_sample = next(iter(train_loader))
x_sample = x_sample.to(device)
lx_sample = lx_sample.to(device)

with torch.no_grad():
    h_sample, lh_sample = model(x_sample, lx_sample)

print(f"Input shape: {x_sample.shape}")
print(f"Input lengths (lx): {lx_sample[:5]}")  # First 5
print(f"Output shape: {h_sample.shape}")
print(f"Output lengths (lh): {lh_sample[:5]}")  # First 5
print(f"Target lengths (ly): {ly_sample[:5]}")  # First 5
print(f"\nMin output length: {lh_sample.min().item()}, Max target length: {ly_sample.max().item()}")
print(f"Problem: Output length < Target length? {(lh_sample.min() < ly_sample.max()).item()}")
print("=" * 40)


# In[39]:


#TODO: Please complete the training loop

for epoch in range(last_epoch_completed, config['epochs']):

    print("\nEpoch: {}/{}".format(epoch + 1, config['epochs']))

    curr_lr = optimizer.param_groups[0]['lr'] #TODO

    train_loss = train_model(model, train_loader, criterion, optimizer) #TODO
    valid_loss, valid_dist = validate_model(model, val_loader, decoder, LABELS) #TODO

    # NOTE: Pass in the appropriate argument based on your chosen scheduler
    #       -> Some schedulers require a metric, while others require the current epoch
    #       -> Some schedulers do not require any argument to be passed in
    scheduler.step()

    print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
    print("\tVal Dist {:.04f}\t Val Loss {:.04f}".format(valid_dist, valid_loss))

    if config['wandb']:
        wandb.log({
            'train_loss': train_loss,
            'valid_dist': valid_dist,
            'valid_loss': valid_loss,
            'lr': curr_lr
    })

    # Save best model
    if valid_dist <= best_lev_dist:
        best_lev_dist = valid_dist
        save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch+1, best_model_path)
        if config['wandb']:
            wandb.save(best_model_path)
        print("Saved best val model")
    
    # Save last epoch model
    save_model(model, optimizer, scheduler, ['valid_dist', best_lev_dist], epoch+1, epoch_model_path)
    if config['wandb']:
        wandb.save(epoch_model_path)
    print("Saved epoch model")

# You may find it interesting to explore Wandb Artifacts to version your models

# Finish Wandb run
if config['wandb']:
    run.finish()


# # Generate Predictions

# In[40]:


# Load best model
print("Loading best model...")
model = load_model(best_model_path, model, optimizer=optimizer, scheduler=scheduler, metric='valid_dist')[0]

torch.cuda.empty_cache()
gc.collect()

# 1. Create decoder with larger beam size for better accuracy at test time
#    (More beams = wider search = better predictions, inference cost doesn't matter as much)
test_decoder = cuda_ctc_decoder(tokens=LABELS, nbest=1, beam_size=config['test_beam_width'])

results = []

model.eval()
print("Testing")

for data in tqdm(test_loader):

    x, lx = data
    x, lx = x.to(device), lx.to(device)

    with torch.no_grad():
        h, lh = model(x, lx)
        
    torch.cuda.synchronize()  # force any CUDA errors from forward pass to surface here
    print("forward pass ok", flush=True)

    # 2. Decode the beam search results into prediction strings
    # Add before calling decode_prediction
    prediction_string = decode_prediction(h, lh.to(device), test_decoder, LABELS)

    # Extend (not append) since decode_prediction returns a list of strings per batch
    results.extend(prediction_string)

    del x, lx, h, lh
    torch.cuda.empty_cache()

print(f"✓ Generated {len(results)} predictions")


# # Submission
# 

# ## Kaggle Submission

# In[43]:


if results:
    df = pd.DataFrame({
        'index': range(len(results)),
        'label': results
    })
# Save submission file
submission_file = "hw3p2_submission.csv"
df.to_csv(submission_file, index=False)
submission_csv = "hw3p2_submission.csv" #TODO: change to the path of your submission file
pd.read_csv(submission_csv).head()


# In[45]:


api.competition_submit(file_name= submission_csv, message="<ENTER_YOUR_MESSAGE_HERE>", competition="11785-hw-3-p-2-utterance-to-phoneme-mapping-spring-2026")
