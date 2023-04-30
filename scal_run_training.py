from utils import tokenize, token_to_index, PAD_IDX
from scal_trainer import train, evaluate
from models import MelodyTransformer
from POP909_parser import parse_songs, special_note_to_indx, PITCH_SUBTRACT_CONSTANT

import argparse
import torch
import torch.nn as nn
import time
import math
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=3)
    parser.add_argument('-train_split', type=float, default=0.8)
    parser.add_argument('-emb_size', type=int, default=512)
    parser.add_argument('-nhead', type=int, default=8)
    parser.add_argument('-hid_dim', type=int, default=512)
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-num_encoder_layers', type=int, default=3)
    parser.add_argument('-num_decoder_layers', type=int, default=3)
    parser.add_argument('-train_split', type=float, default=0.8)
    args = parser.parse_args()

    # Source is a set of chords with the structure tokens filtered out
    # Thus, major and minor versions of 1-7, 14
    args.src_vocab_size = 14

    # num possible pitches + Rest
    args.tgt_vocab_size = 200 # TBD
    # Getting data: 
    POP_path = '/Users/benshi/Documents/BillboardProject/ChordTransformer/POP909'
    songs = parse_songs(POP_path)

    inputs = []
    targets = []

    def query_note(chord_seq, start_time):
        for chord in chord_seq:
            if chord.end_time > start_time:
                return chord
        return None

    # Processing for same tempo unit
    # Truncate chords to only count chords that occur after the melody starts
    # BASICALLY WHAT WE WANT TO DO: FOR EACH NOTE TOKEN ADDED, WE LOOK UP ITS CORRESPONDING CHORD
    for song in songs:
        counter = 1
        notes = []
        chords = []
        for note in song.notes:
            diff = note.end - note.start
            units = round(diff / song.tempo_unit)
            for i in range(units):
                if i == 0:
                    corr_chord = query_note(song.chord_seq, note.start)
                    if corr_chord == None: 
                        continue 
                    notes.append(note.pitch)
                    chords.append(token_to_index[corr_chord.get_chord_string()])
                else:
                    corr_chord = query_note(song.chord_seq, note.start + i*song.tempo_unit)
                    if corr_chord == None: 
                        continue 
                    notes.append(special_note_to_indx['Hold'])
                    chords.append(token_to_index[corr_chord.get_chord_string()])
            
            if counter != len(song.notes):
                next_note = song.notes[counter]
                rest_time = next_note.start - note.end
                if rest_time > 0.8 * song.tempo_unit:
                    units = round(rest_time / song.tempo_unit)
                    for i in range(units):
                        corr_chord = query_note(song.chord_seq, note.end + i*song.tempo_unit)
                        if corr_chord == None: 
                            continue
                        notes.append(special_note_to_indx['Rest'])
                        chords.append(token_to_index[corr_chord.get_chord_string()])
            counter += 1
        
        assert len(notes) == len(chords)
        for i in range(0, len(notes), 100):
            if len(notes) - i > 100:
                inputs.append(chords[i:i+100])
                targets.append(notes[i:i+100])
    
    train_split_indx = int(len(inputs)*args.train_split)
    train_data_inputs = inputs[:train_split_indx]
    train_data_targets = targets[:train_split_indx]
    val_split_indx = train_split_indx + int(len(inputs)*((1-args.train_split)/2))
    val_data_inputs = inputs[train_split_indx:val_split_indx]
    val_data_targets = targets[train_split_indx:val_split_indx]
    test_data_inputs = inputs[val_split_indx:]
    test_data_targets = targets[val_split_indx:]

    class EDArgs:
        def __init__(self, d_model, d_hid, nhead, nlayers):
            self.d_model = d_model
            self.d_hid = d_hid
            self.nhead = nhead
            self.nlayers = nlayers

    encoder_args = EDArgs(args.emb_size, args.hid_dim, args.nhead, args.num_encoder_layers)
    decoder_args = EDArgs(args.emb_size, args.hid_dim, args.nhead, args.num_decoder_layers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MelodyTransformer(encoder_args, decoder_args, args.src_vocab_size, args.tgt_vocab_size)

    torch.manual_seed(0)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    model = model.to(device)
    args.train_inputs = train_data_inputs
    args.train_targets = train_data_targets
    args.criteron = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    args.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    args.scheduler = torch.optim.lr_scheduler.StepLR(args.optimizer, 1.0, gamma=0.95)

    best_val_loss = float('inf')
    epochs = args.epochs
    temp_path = '/Users/benshi/Documents/BillboardProject/ChordTransformer/models/chord_gen.pt'
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train(model, args)
        val_loss = evaluate(model, args, val_data_inputs, val_data_targets)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-'*89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | 'f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), temp_path)

        args.scheduler.step()

    model.load_state_dict(torch.load(temp_path))

    print('FINAL EVALUATION')
    test_loss = evaluate(model, args, test_data_inputs, test_data_targets)
    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f} | 'f'test ppl {test_ppl:8.2f}')
    print('=' * 89)

    torch.save(model, args.output_dir)

if __name__ == '__main__':
    main()