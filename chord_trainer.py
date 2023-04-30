from utils import generate_square_subsequent_mask, get_input_output, get_input_output_POP, PAD_IDX

import math 
import time 
import torch

def train(model, args):
    print("Training Model...")
    model.train()
    start_time = time.time()
    total_loss = 0
    log_interval = 10
    src_mask = generate_square_subsequent_mask(args.bptt).to(args.device)

    for i in range(0, len(args.train)):
        if args.data_type == 'guitartabs':
            data, targets = get_input_output(args.train[i])
        elif args.data_type == 'POP909':
            data, targets = get_input_output_POP(args.train[i])
        src_key_padding_mask = data != PAD_IDX
        src_key_padding_mask = src_key_padding_mask.view(-1)
        output = model(data, src_mask)
        loss = args.criterion(output.view(-1, args.ntokens), targets)
        loss_masked = loss.where(src_key_padding_mask, torch.tensor(0.0)).mean()
        args.optimizer.zero_grad()
        loss_masked.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        args.optimizer.step()

        total_loss += loss_masked.item()

        if i % log_interval == 0 and i > 0:
            lr = args.scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | 'f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()
    #for i, batch in enumerate(args.train):

def evaluate(model, args, eval_data):
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(args.bptt).to(args.device)
    with torch.no_grad():
        for i in range(0, len(eval_data)):
            if args.data_type == 'guitartabs':
                data, targets = get_input_output(eval_data[i])
            elif args.data_type == 'POP909':
                data, targets = get_input_output_POP(eval_data[i])
            src_key_padding_mask = data != PAD_IDX
            src_key_padding_mask = src_key_padding_mask.view(-1)
            output = model(data, src_mask)
            output_flat = output.view(-1, args.ntokens)
            loss = args.criterion(output_flat, targets)
            loss_masked = loss.where(src_key_padding_mask, torch.tensor(0.0)).mean()
            total_loss += loss_masked.item()
    return total_loss / (len(eval_data))
    



