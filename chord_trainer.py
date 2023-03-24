from utils import generate_square_subsequent_mask, get_input_output

import math 
import time 
import torch

def train(model, args):
    print("Training Model...")
    model.train()
    start = time.time()
    total_loss = 0
    log_interval = 10
    src_mask = generate_square_subsequent_mask(args.bptt).to(args.device)

    for i in range(0, len(args.train)):
        data, targets = get_input_output(args.train[i])
        seq_len = data.size(0)
        output = model(data, src_mask)
        loss = args.criterion(output.view(-1, args.ntokens), targets)
        
        args.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        args.optimizer.step()

        total_loss += loss.item()

        if i % log_interval == 0:
            lr = args.scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | 'f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | 'f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

    for epoch in range(args.epochs):
        total_loss = 0
        
        if args.step_metrics:
            print("   %dm: epoch %d [%s]  %d%%  loss = %s" % ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
        
        if args.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')
    
    #for i, batch in enumerate(args.train):

def evaluate(model, args, eval_data):
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(args.bptt).to(args.device)
    with torch.no_grad():
        for i in range(0, len(eval_data)):
            data, targets = get_input_output(eval_data[i])
            seq_len = data.size(0)
            output = model(data, src_mask)
            output_flat = output.view(-1, args.ntokens)
            total_loss += seq_len * args.criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)
    



