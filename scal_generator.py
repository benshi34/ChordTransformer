import torch
import torch.nn as nn
import numpy as np
from utils import token_to_index, PAD_IDX, generate_square_subsequent_mask


def main():
    path = '/Users/benshi/Documents/BillboardProject/ChordTransformer/models/melody_generator'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path, device).to(device)
    chord_generator = MelodyGenerator(model, device)
    temperature = 0.8

    context_seq = 'Start-Verse-1M'
    context_seq = context_seq.split('-')
    context_seq_idx = []
    for token in context_seq:
        context_seq_idx.append(token_to_index[token])
    
    # Call different functions to generate differently
    output = chord_generator.generate_all_p_sampling(context_seq_idx, temperature)
    print(output)

class MelodyGenerator:
    def __init__(self, model, device):
        self.model = model
        self.max_len = 222
        self.src_mask = generate_square_subsequent_mask(self.max_len).to(device)
        self.softmax = nn.Softmax()

    def generate_next_greedy(self, context_seq, temperature=1.0):
        with torch.no_grad():
            next_idx = len(context_seq)
            seq = context_seq.copy()
            while len(seq) < self.max_len:
                seq.append(PAD_IDX)
            seq = torch.tensor(seq).view(-1, 1)
            output = self.model(seq, self.src_mask)
            if temperature != 1.0:
                output = output / temperature
            output = self.softmax(output)
            output = output.view(-1, len(token_to_index))
            output = torch.argmax(output[next_idx])
        return output

    def generate_all_greedy(self, context_seq, temperature=1.0):
        with torch.no_grad():
            seq = context_seq.copy() 
            curr_len = len(seq)
            while len(seq) < self.max_len:
                seq.append(PAD_IDX)
            while curr_len < self.max_len: 
                curr_seq = torch.tensor(seq).view(-1, 1)
                output = self.model(curr_seq, self.src_mask)
                if temperature != 1.0:
                    output = output / temperature
                output = self.softmax(output)
                output = output.view(-1, len(token_to_index))
                output = torch.argmax(output[curr_len-1]).item()
                seq[curr_len] = output
                curr_len += 1
            return seq
    
    def generate_all_k_sampling(self, context_seq, temperature=1.0, k=8):
        with torch.no_grad():
            seq = context_seq.copy() 
            curr_len = len(seq)
            while len(seq) < self.max_len:
                seq.append(PAD_IDX)
            while curr_len < self.max_len: 
                curr_seq = torch.tensor(seq).view(-1, 1)
                output = self.model(curr_seq, self.src_mask)
                if temperature != 1.0:
                    output = output / temperature
                output = self.softmax(output)
                output = output.view(-1, len(token_to_index))
                output = torch.topk(output[curr_len-1], k)

                # Change so that probabilities sum to 1:
                probabilities = output.values / torch.sum(output.values)
                next_token_idx = np.random.multinomial(1, list(probabilities.tolist()))
                next_token_idx = list(next_token_idx).index(1)
                next_token = output.indices[next_token_idx].item()
                seq[curr_len] = next_token
                curr_len += 1
            return seq
    
    def generate_all_p_sampling(self, context_seq, temperature=1.0, p=0.8):
        with torch.no_grad():
            seq = context_seq.copy()
            curr_len = len(seq)

            while len(seq) < self.max_len:
                seq.append(PAD_IDX)
            while curr_len < self.max_len: 
                curr_seq = torch.tensor(seq).view(-1, 1)
                output = self.model(curr_seq, self.src_mask)
                if temperature != 1.0:
                    output = output / temperature
                output = self.softmax(output)
                output = output.view(-1, len(token_to_index))
                output = torch.sort(output[curr_len-1], descending=True)
                output = output.tolist()
                
                index = 0
                sum = 0
                while sum < p:
                    sum += output[index]
                    index += 1
                output = output[:index]
                # Change so that probabilities sum to 1:
                next_token_idx = np.random.multinomial(1, output)
                next_token_idx = list(next_token_idx).index(1)
                next_token = output.indices[next_token_idx].item()
                seq[curr_len] = next_token
                curr_len += 1
            return seq

if __name__ == '__main__':
    main()