import time 
import torch
import dill as pickle
from utils import generate_square_subsequent_mask

def train(model, args):
    print("Training Model...")
    model.train()
    start = time.time()
    total_loss = 0
    log_interval = 100
    src_mask = generate_square_subsequent_mask(args.bptt).to(args.device)

    num_batches = len(args.train) // args.bptt
    for epoch in range(args.epochs):
        total_loss = 0
        
        if args.step_metrics:
            print("   %dm: epoch %d [%s]  %d%%  loss = %s" % ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
        
        if args.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')
    
    #for i, batch in enumerate(args.train):

def evaluate(model, args):
    model.eval()
    total_loss = 0
    src_mask = generate_square_subsequent_mask(args.bptt).to(args.device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, args.bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            if seq_len != args.bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, args.ntokens)
            total_loss += seq_len * args.criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)
    



