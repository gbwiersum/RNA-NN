#!/usr/bin/env python3

import gzip
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from Bio import SeqIO
import torch


def gpu_status():
    gpu_present = [torch.cuda.device(i) for i in range(torch.cuda.device_count())] != []
    print("GPU Detected?: " + str(gpu_present))
    if gpu_present:
        torch.set_default_device('cuda')
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device


def get_seqios(file):
    seqs = {}
    if file[-2::] == "gz":
        with gzip.open(file, 'rt') as fastq:
            for index, record in enumerate(SeqIO.parse(fastq, 'fastq')):
                seqs.update({record.id:str(record.seq)})
            return seqs

    else:
        with open(file, "r") as fastq:
            for index, record in enumerate(SeqIO.parse(fastq, 'fastq')):
                seqs.update({record.id:str(record.seq)})
            return seqs


def parse_reads(record, pad_size):
    enc = OneHotEncoder()
    enc.fit(np.array(["A", "T", "C", "G", "N"]).reshape(-1, 1))
    x_in = np.array(list(record))
    arr = enc.fit_transform(x_in.reshape(-1, 1)).toarray()
    delta = len(arr) - pad_size

    if delta > 0:
        # random crop
        shift = np.random.randint(0, delta)
        x_out = arr[shift:shift + pad_size]

    else:
        arr.resize((pad_size, 4), refcheck=False)
        x_out = arr

    return x_out
