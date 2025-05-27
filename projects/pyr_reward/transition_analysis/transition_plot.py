"""
plt transition data
"""
import pickle, os, numpy as np

fl = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\TransitionResults\TransitionMatrix\NeuronType"
with open(fl, "rb") as fp: #unpickle
    dct = pickle.load(fp)
