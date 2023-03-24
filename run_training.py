from utils import tokenize
from trainer import train, evaluate
from models import Transformer
from tempfile import TemporaryDirectory
import argparse
import torch
import torch.nn as nn
import os
import time
import math
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', required=True)
    parser.add_argument('-step_metrics', type=bool, default=False)
    parser.add_argument('-learning_rate', type=int, default=0.2)
    parser.add_argument('-bptt', type=int, default=1) # Length to subdivide source data
    parser.add_argument('-d_hid', type=int, default=50) # dimension of the hidden layer
    parser.add_argument('-emsize', type=int, default=30) # Embedding size
    parser.add_argument('-dropout', type=float, default=0.2) # Dropout
    parser.add_argument('-train_split', type=float, default=0.8)
    args = parser.parse_args()

    # Checking if GPU is available
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Processing data
    file = open('progressions.txt', 'r')
    progression_data = file.read()
    songs = progression_data.split('\n\n')
    
    # TOKENIZING
    data = tokenize(songs)
    np.random.shuffle(data)

    # Train test val split: Does not account for edge case of low data points but this should not be issue 
    train_split_indx = int(len(data)*args.train_split)
    train_data = data[:train_split_indx]
    val_split_indx = train_split_indx + int(len(data)*((1-args.train_split)/2))
    val_data = data[train_split_indx:val_split_indx]
    test_data = data[val_split_indx:]

    args.ntokens = 21 # Predetermined
    args.train = train_data
    args.eval = val_data
    args.test = test_data
    args.optimizer = torch.optim.SGD(args.model.parameters(), lr=args.learning_rate) # Setting learning rate
    args.criteron = nn.CrossEntropyLoss()
    args.scheduler = torch.optim.lr_scheduler.StepLR(args.optimizer, 1.0, gamma=0.95)
    model = Transformer(args.ntokens, args.emsize, args.nhead, args.d_hid, args.nlayers, args.dropout).to(args.device)

    best_val_loss = float('inf')
    epochs = 3

    with TemporaryDirectory() as tempdir: 
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")
         
        for epoch in range(1, epochs+1):
            epoch_start_time = time.time()
            train(model)
            val_loss = evaluate(model, args, val_data)
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - epoch_start_time
            print('-'*89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | 'f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)

            args.scheduler.step()
        
        model.load_state_dict(torch.load(best_model_params_path))

    print('FINAL EVALUATION')
    test_loss = evaluate(model, args, test_data)
    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f} | 'f'test ppl {test_ppl:8.2f}')
    print('=' * 89)

if __name__ == '__main__':
    main()