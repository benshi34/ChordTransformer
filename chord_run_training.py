from utils import tokenize, token_to_index, PAD_IDX
from chord_trainer import train, evaluate
from models import Transformer
from POP909_parser import parse_chord_sequence

import argparse
import torch
import torch.nn as nn
import time
import math
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-step_metrics', type=bool, default=False)
    parser.add_argument('-learning_rate', type=int, default=0.05)
    parser.add_argument('-d_hid', type=int, default=50) # dimension of the hidden layer
    parser.add_argument('-emsize', type=int, default=20) # Embedding size
    parser.add_argument('-dropout', type=float, default=0.2) # Dropout
    parser.add_argument('-nhead', type=int, default=4) # Number of attention heads
    parser.add_argument('-nlayers', type=int, default=2) # Number of transformer encoder layers
    parser.add_argument('-train_split', type=float, default=0.8)
    parser.add_argument('-data_type', type=str, default='guitartabs')
    parser.add_argument('-output_dir', type=str, default='/Users/benshi/Documents/BillboardProject/ChordTransformer/models/chord_generator_POP909')
    args = parser.parse_args()

    # Checking if GPU is available
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert args.data_type in {'guitartabs', 'POP909'}
    if args.data_type == 'guitartabs':
        # Processing data
        file = open('progressions.txt', 'r')
        progression_data = file.read()
        songs = progression_data.split('\n\n')
        
        # TOKENIZING
        data = tokenize(songs)
        args.bptt = len(data[0].data)
    
    elif args.data_type == 'POP909':
        POP_path = '/Users/benshi/Documents/BillboardProject/ChordTransformer/POP909'
        chord_sequences = parse_chord_sequence(POP_path)
        data = []
        args.bptt = -1
        # Each song is a set of chord sequences
        for song in chord_sequences:
            # Figuring out basic tempo unit:
            tempo_unit = float('inf')
            for chord in song: 
                diff = float(chord.end_time) - float(chord.start_time)
                tempo_unit = min(tempo_unit, diff)
            
            processed_chord_seq = []
            for chord in song:
                diff = float(chord.end_time) - float(chord.start_time)
                units = round(diff / tempo_unit)
                for _ in range(units):
                    string = str(chord.modified_root) + str(chord.modified_quality)
                    processed_chord_seq.append(token_to_index[string])
            
            args.bptt = max(args.bptt, len(processed_chord_seq))
            data.append(processed_chord_seq)
        
        for song in data:
            while len(song) < args.bptt:
                song.append(token_to_index['Pad'])
    
    np.random.shuffle(data)
    # Train test val split: Does not account for edge case of low data points but this should not be issue 
    train_split_indx = int(len(data)*args.train_split)
    train_data = data[:train_split_indx]
    val_split_indx = train_split_indx + int(len(data)*((1-args.train_split)/2))
    val_data = data[train_split_indx:val_split_indx]
    test_data = data[val_split_indx:]

    args.ntokens = len(token_to_index) # Predetermined
    args.train = train_data
    args.eval = val_data
    args.test = test_data
    model = Transformer(args.ntokens, args.emsize, args.nhead, args.d_hid, args.nlayers, args.dropout).to(args.device)
    args.optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate) # Setting learning rate
    args.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction='none')
    args.scheduler = torch.optim.lr_scheduler.StepLR(args.optimizer, 1.0, gamma=0.95)

    best_val_loss = float('inf')
    epochs = args.epochs
    
    temp_path = '/Users/benshi/Documents/BillboardProject/ChordTransformer/models/chord_gen.pt'
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train(model, args)
        val_loss = evaluate(model, args, val_data)
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
    test_loss = evaluate(model, args, test_data)
    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f} | 'f'test ppl {test_ppl:8.2f}')
    print('=' * 89)

    torch.save(model, args.output_dir)

if __name__ == '__main__':
    main()