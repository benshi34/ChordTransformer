from utils import Progression, PPMI, create_dataset
from trainer import train
from models import Transformer
import argparse
import torch
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', required=True)
    parser.add_argument('-step_metrics', type=bool, default=False)
    parser.add_argument('-learning_rate', type=int, default=0.2)
    parser.add_argument('-bptt', type=int, default=40) # Length to subdivide source data
    parser.add_argument('-d_hid', type=int, default=50) # dimension of the hidden layer
    parser.add_argument('-emsize', type=int, default=30) # Embedding size
    parser.add_argument('-dropout', type=float, default=0.2) # Dropout
    args = parser.parse_args()

    # Checking if GPU is available
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Processing data
    file = open('progressions.txt', 'r')
    progression_data = file.read()
    songs = progression_data.split('\n\n')

    prog_arr = []
    for song in songs:
        prog_arr.append(Progression(song))

    args.ntokens = 19 # Predetermined
    args.train = prog_arr
    args.model = Transformer(args.ntokens, args.emsize, args.nhead, args.d_hid, args.nlayers, args.dropout).to(args.device)
    args.optimizer = torch.optim.SGD(args.model.parameters(), lr=args.learning_rate) # Setting learning rate
    args.criteron = nn.CrossEntropyLoss()
    args.scheduler = torch.optim.lr_scheduler.StepLR(args.optimizer, 1.0, gamma=0.95)

    # Context window for PPMI: Using 4 to the left and 4 to the right since majority are 4 chord progressions
    context_window = (4, 4)
    ppmi = PPMI(context_window)
    for prog in prog_arr:
        ppmi.addProgression(prog)
    ppmi_embeddings = ppmi.genPPMI()
    print(ppmi_embeddings)

    best_val_loss = float('inf')
    epochs = 3

if __name__ == '__main__':
    main()