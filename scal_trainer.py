from utils import generate_square_subsequent_mask, get_input_output, get_input_output_POP, PAD_IDX

import torch
import time
import math

def train(model, args):
    print("Training Model...")
    model.train()
    start_time = time.time()
    total_loss = 0
    log_interval = 10
    src_mask = generate_square_subsequent_mask(args.bptt).to(args.device)
    tgt_mask = generate_square_subsequent_mask(args.bptt).to(args.device)

    input_ids = args.train_inputs
    target_ids = args.train_targets
    for i in range(0, len(input_ids)):
        src_key_padding_mask = input_ids != PAD_IDX
        src_key_padding_mask = src_key_padding_mask.view(-1)
        tgt_key_padding_mask = target_ids != PAD_IDX
        tgt_key_padding_mask = tgt_key_padding_mask.view(-1)
        output = model(input_ids, src_mask, target_ids, tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        loss = args.criterion(output.view(-1, args.tgt_vocab_size), target_ids)
        loss_masked = loss.where(tgt_key_padding_mask, torch.tensor(0.0)).mean()
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

def evaluate(model, args, val_data_inputs, val_data_targets):
    model.eval()  # turn on evaluation mode
    total_loss = 0
    src_mask = generate_square_subsequent_mask(args.bptt).to(args.device)
    tgt_mask = generate_square_subsequent_mask(args.bptt).to(args.device)

    with torch.no_grad():
        for i in range(0, len(val_data_inputs)):
            src_key_padding_mask = val_data_inputs != PAD_IDX
            src_key_padding_mask = src_key_padding_mask.view(-1)
            tgt_key_padding_mask = val_data_targets != PAD_IDX
            tgt_key_padding_mask = tgt_key_padding_mask.view(-1)
            
            output = model(val_data_inputs, src_mask, val_data_targets, tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            output_flat = output.view(-1, args.tgt_vocab_size)
            loss = args.criterion(output_flat, val_data_targets)
            loss_masked = loss.where(tgt_key_padding_mask, torch.tensor(0.0)).mean()
            total_loss += loss_masked.item()
    
    return total_loss / (len(val_data_inputs))



