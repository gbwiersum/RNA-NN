#!/usr/bin/env python3
import glob
import torch
from torch.utils import data
import utils
import re
# Todo: replace pandas with numpy
import pandas as pd

class AE(torch.nn.Module):
    def __init__(self, pad_size):
        super().__init__()
        self.PAD_SIZE = pad_size

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(self.PAD_SIZE, 800, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(800, 400, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(400, 128, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 64, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 16, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 16, 1)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Conv1d(16, 16, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 64, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 400, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(400, 800, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(800, self.PAD_SIZE, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class mRNA_Encoder:

    def __init__(self, model, pad_size=1000, checkpoint=None):
        self.pad_size = pad_size
        self.model = model
        self.device = utils.gpu_status()
        if checkpoint is not None:
            state_dict = torch.load(checkpoint)
            self.model.load_state_dict(state_dict=state_dict)

    class Dataset(data.Dataset):
        def __init__(self, training_path, pad_size):
            self.data = glob.glob(training_path)
            self.PAD_SIZE = pad_size

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            path = self.data[idx]
            out = utils.parse_reads(path, self.PAD_SIZE)
            out = out.reshape(self.PAD_SIZE, 4)
            out = torch.tensor(out).to(torch.float)
            return out

    def train_encoder(self, training_path, epochs=1, pad_size=1000, batch_size=1, lr=0.1, decay=1e-9):
        dataset = self.Dataset(training_path, pad_size)
        epochs = epochs
        batch_size = batch_size
        lr = lr
        decay = decay
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=0,
                                                 generator=torch.Generator(device='cuda'))

        losses = []
        for epoch in range(epochs):
            print("epoch" + str(epoch))
            batch = next(iter(dataloader))
            for batch_index, doc in enumerate(batch):
                recon = self.model(doc)
                # Loss function
                loss = loss_function(recon, doc)
                if batch_index % 10 == 0:
                    print("Batch: " + str(batch_index))
                    print("loss" + str(loss))

                # Gradients are set to zero,
                # Gradient is computed and stored.
                # .step() performs parameter update.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss)

        return self.model, losses

    def seq_to_embedding(self, seq):
        seq = utils.parse_reads(seq, pad_size=self.pad_size)
        seq = torch.tensor(seq).to(torch.float)
        embedding = self.model.encoder(seq).cpu().detach().numpy().flatten()
        return embedding

    def embed_file(self, file):
        seqs = utils.get_seqios(file)
        df = pd.DataFrame.from_dict(seqs, orient='index')
        with torch.no_grad():
            sample = df.apply(self.seq_to_embedding, axis=1, result_type='expand')
        return sample

    def embed_fastqs(self, in_filepath="", out_filepath=""):
        fastqs = glob.glob(in_filepath + "*.fastq.gz")
        for n, file in enumerate(fastqs):
            subbed = re.sub("fastq", "encoded", file)
            subbed = re.sub(".fastq.gz", ".csv", subbed)
            encoded = self.embed_file(file)
            encoded.to_csv(out_filepath + subbed)
            print("Finished: " + str(n) + " Of " + str(len(fastqs)))