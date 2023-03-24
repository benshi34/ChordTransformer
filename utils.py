import torch
import torch.nn as nn
import math

token_to_index = {'1m':0, '1M':1, '2m':2, '2M':3, '3m':4, '3M':5, '4m':6, '4M':7, '5m':8, '5M':9, '6m':10, '6M':11, '7m':12, '7M':13, 'Intro':14, 'Verse':15, 'Prechorus':16, 'Chorus':17, 'Start':18, 'End':19, 'Pad':20}

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Progression:
    def __init__(self, progression_str):
        data = progression_str.split('\n')
        self.song_name = data[0][:-1]
        data.pop(0)
        
        prog = '-'.join(data)
        prog = prog.replace(':', '')
        self.data = prog.split('-')
        
    def __str__(self):
        return str(self.data)

# EXTRA
class PPMI:
    def __init__(self, context_window):
        matrix = []
        for i in range(19):
            matrix.append([0]*19)
            
        # Each row is the PMI vector representation of token i
        self.matrix = matrix
        self.context_left, self.context_right = context_window
        self.token_to_index = {'1m':0, '1M':1, '2m':2, '2M':3, '3m':4, '3M':5, '4m':6, '4M':7, '5m':8, '5M':9, '6m':10, '6M':11, '7m':12, '7M':13, 'Intro':14, 'Verse':15, 'Prechorus':16, 'Chorus':17, 'Start':18, 'End':19, 'Pad':20}
    # Takes in a progression and handles the counting
    def addProgression(self, progression):
        prog = progression.data
        
        # Filter out empty progressions if they got in
        if len(prog) <= 1:
            return
        for i in range(len(prog)):
            left_index = max(0, i - self.context_left)
            right_index = min(len(prog)-1, i + self.context_right)
            matrix_index = self.token_to_index[prog[i]]
            for j in range(left_index, right_index+1):
                if j == i:
                    continue
                token_index = self.token_to_index[prog[j]]
                self.matrix[matrix_index][token_index] += 1
    def genPPMI(self):
        ppmi = []
        for i in range(len(self.matrix)):
            ppmi.append([0]*19)
        
        totalSum, rowSums = self._sums()
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                joint_prob = self.matrix[i][j] / totalSum
                prob_i = rowSums[i] / totalSum
                prob_j = rowSums[j] / totalSum
                
                if prob_i != 0 and prob_j != 0 and joint_prob != 0:
                    ppmi_val = max(0, math.log2(joint_prob / prob_i / prob_j))
                    ppmi[i][j] = ppmi_val
        
        return ppmi
    def _sums(self):
        total = 0
        rowSums = []
        for row in self.matrix:
            tempSum = sum(row)
            total += tempSum
            rowSums.append(tempSum)
        return total, rowSums
    def __str__(self):
        for row in self.matrix: 
            print(row)

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# Input: Array of progression
# Output: Tensor of data and target
def get_input_output(prog):
    print(prog.data)
    data = torch.tensor(prog.data)
    target = prog.data[1:]
    target.append(20) # Int from of pad token
    target = torch.tensor(target)
    return data, target

def tokenize(songs):
    prog_arr = []
    for song in songs:
        prog_arr.append(Progression(song))
    
    # Adding end tokens
    for prog in prog_arr: 
        prog.data.append('End')
        prog.data.insert(0, 'Start')
    
    # Padding to all the same length
    maxi = 0
    for prog in prog_arr: 
        maxi = max(maxi, len(prog.data))
    for prog in prog_arr:
        while len(prog.data) < maxi: 
            prog.data.append('Pad')

    # Changing strings to int representation
    for prog in prog_arr: 
        for i in range(len(prog.data)):
            prog.data[i] = token_to_index[prog.data[i]]

    return prog_arr