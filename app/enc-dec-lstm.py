import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

import random
import math
import time
from tqdm import tqdm

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src len, batch size, hid dim * n directions] # if bidirectional
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer
        return outputs, hidden, cell
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim) + dec_hid_dim, dec_hid_dim) # If encoder is bidirectional, use enc_hid_dim * 2
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [1, batch size, dec hid dim] # If decoder is single layer
        # encoder_outputs = [src len, batch size, enc hid dim] # If encoder is single layer, use enc_hid_dim * 2 if bidirectional

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # Repeat decoder hidden state src_len times
        hidden = hidden.squeeze(0).unsqueeze(0).repeat(src_len, 1, 1)
        # hidden = [src len, batch size, dec hid dim]

        # Calculate energy between hidden and encoder outputs
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [src len, batch size, dec hid dim]

        # Calculate attention weights
        attention = self.v(energy).squeeze(2)
        # attention = [src len, batch size]

        return torch.softmax(attention, dim=0)

class DecoderRNN(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, dropout=dropout) # If encoder is bidirectional, use hid_dim * 2
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim) # If encoder is bidirectional, use hid_dim * 3
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input = [batch size]
        # hidden = [1, batch size, dec hid dim]
        # cell = [1, batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim]

        input = input.unsqueeze(0)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]

        # Calculate attention weights
        # attention_weights = [src len, batch size]
        attention_weights = self.attention(hidden, encoder_outputs)

        # --- FIX START ---
        # Permute attention weights to have batch_size first: [batch size, src len]
        attention_weights = attention_weights.permute(1, 0)
        # attention_weights = [batch size, src len]

        # Unsqueeze to add a dimension for matrix multiplication: [batch size, 1, src len]
        attention_weights = attention_weights.unsqueeze(1)
        # attention_weights = [batch size, 1, src len]

        # Permute encoder_outputs to have batch_size first: [batch size, src len, enc hid dim]
        # This line was already correct for bmm
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src len, enc hid dim]

        # Perform batch matrix multiplication: (batch size, 1, src len) x (batch size, src len, enc hid dim)
        weighted_context = torch.bmm(attention_weights, encoder_outputs).squeeze(1)
        # weighted_context = [batch size, enc hid dim]
        # --- FIX END ---

        weighted_context = weighted_context.unsqueeze(0)
        # weighted_context = [1, batch size, enc hid dim]

        rnn_input = torch.cat((embedded, weighted_context), dim=2)
        # rnn_input = [1, batch size, emb dim + enc hid dim]

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # output = [1, batch size, hid dim]
        # hidden = [1, batch size, hid dim]
        # cell = [1, batch size, hid dim]

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_context = weighted_context.squeeze(0) # Squeeze again after unsqueeze(0) for concat

        prediction = self.fc_out(torch.cat((output, weighted_context, embedded), dim=1))
        # prediction = [batch size, output dim]

        # Return attention weights in a more usable shape if needed, e.g., [batch size, src len]
        # Let's return the shape after the first permute: [batch size, src len]
        return prediction, hidden, cell, attention_weights.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.rnn.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use ground-truth target word

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # Encoder outputs, and final hidden and cell states
        encoder_outputs, hidden, cell = self.encoder(src)

        # First input to the decoder is the <sos> token
        input = trg[0, :]

        for t in range(1, trg_len):
            # Get output prediction from decoder
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs)

            # Store prediction
            outputs[t] = output

            # Decide if we will use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio

            # Get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # If teacher forcing, use actual next token, otherwise use predicted token
            input = trg[t, :] if teacher_force else top1

        return outputs

def load_cnn_dailymail_dataset(num_train_samples=None, num_val_samples=None):
    # Using Hugging Face datasets for easy access
    print("Loading CNN/DailyMail dataset...")
    dataset = load_dataset("cnn_dailymail", '3.0.0') # Specify version

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # Or a summarization-specific tokenizer

    # Add special tokens for start and end of sequence
    tokenizer.add_special_tokens({'bos_token': '<sos>', 'eos_token': '<eos>'})

    def tokenize_function(examples):
        # Tokenize articles
        model_inputs = tokenizer(examples['article'], max_length=512, truncation=True, padding="max_length")
        # Tokenize highlights (summaries)
        labels = tokenizer(examples['highlights'], max_length=128, truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Select a subset of data if specified
    if num_train_samples is not None:
        tokenized_datasets['train'] = tokenized_datasets['train'].select(range(num_train_samples))
        print(f"Using {len(tokenized_datasets['train'])} training samples.")
    if num_val_samples is not None:
        tokenized_datasets['validation'] = tokenized_datasets['validation'].select(range(num_val_samples))
        print(f"Using {len(tokenized_datasets['validation'])} validation samples.")

    # Remove original text columns
    tokenized_datasets = tokenized_datasets.remove_columns(["article", "highlights", "id"])
    tokenized_datasets.set_format("torch")

    return tokenized_datasets, tokenizer

# Training Function
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for batch in tqdm(iterator, desc="Training"):
        src = batch['input_ids'].transpose(0, 1).to(model.device) # [src len, batch size]
        trg = batch['labels'].transpose(0, 1).to(model.device)   # [trg len, batch size]

        optimizer.zero_grad()

        output = model(src, trg)
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        # Reshape for criterion: (N, C) where N is number of predictions, C is number of classes
        # Use .reshape() instead of .view()
        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Evaluation Function
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):
            src = batch['input_ids'].transpose(0, 1).to(model.device) # [src len, batch size]
            trg = batch['labels'].transpose(0, 1).to(model.device)   # [trg len, batch size]

            output = model(src, trg, 0) # Turn off teacher forcing for evaluation
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            # Use .reshape() instead of .view()
            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Configuration ---
    NUM_TRAIN = 5000 # Set your desired number of training samples
    NUM_VAL = 1000    # Set your desired number of validation samples
    BATCH_SIZE = 32
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    N_EPOCHS = 20
    CLIP = 1.0

    # Load and preprocess data
    tokenized_datasets, tokenizer = load_cnn_dailymail_dataset(num_train_samples=NUM_TRAIN, num_val_samples=NUM_VAL)

    # Create DataLoaders
    train_iterator = DataLoader(tokenized_datasets['train'], batch_size=BATCH_SIZE, shuffle=True)
    valid_iterator = DataLoader(tokenized_datasets['validation'], batch_size=BATCH_SIZE)

    INPUT_DIM = len(tokenizer)
    OUTPUT_DIM = len(tokenizer)
    TRG_PAD_IDX = tokenizer.pad_token_id

    # Initialize model components
    attention = Attention(HID_DIM, HID_DIM) # Use HID_DIM for both if encoder is not bidirectional
    encoder = EncoderRNN(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
    decoder = DecoderRNN(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT, attention)

    model = Seq2Seq(encoder, decoder, device).to(device)

    # Initialize weights
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

    model.apply(init_weights)

    # Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX) # Ignore padding index in loss calculation

    best_valid_loss = float('inf')

    # Training Loop
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):.3f}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_seq2seq_attn_model.pt')

    print("Training finished.")

if __name__=="__main__":
    main()