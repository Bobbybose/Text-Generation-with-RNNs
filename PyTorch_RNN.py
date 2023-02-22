# Author: Bobby Bose
# Assignment 3: Text Generation with RNNs
# Note: Based off of example by Dr. Brent Harrison showed during class

import numpy as np
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 33

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        # Storing network parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Creating RNN
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        # Output (fully connected) layer
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # Initializing hidden state
        # 1 is for batch size
        hidden_state = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size)
        hidden_state = hidden_state.to(device)
        
        # Running input through RNN
        if (len(x) == BATCH_SIZE):
            output, hidden_state = self.rnn(x, hidden_state)

        # Reforming because not doing batches
        #output = output.contiguous().view(-1, self.hidden_size)

        # Receiving output from fully connected layer
        output = self.fc(output)

        return output, hidden_state


def main():    

    # Reading in training data
    tiny_shakespeare = open('tiny-shakespeare.txt', 'r')
    #sentences = tiny_shakespeare.readlines()
    sentences = [line for line in tiny_shakespeare.readlines() if line.strip()]

    # Making a list of characters in training data
    characters = set(''.join(sentences))

    # Map index to character
    intToChar = dict(enumerate(characters))

    # Map character to index
    charToInt = {character: index for index, character in intToChar.items()}

    # Obtaining input and output sequences
    input_sequence = []
    target_sequence = []
    for i in range(len(sentences)):
        input_sequence.append(sentences[i][:-1])
        target_sequence.append(sentences[i][1:])

    # Replacing all characters with associated integer
    for i in range(len(sentences)):
        input_sequence[i] = [charToInt[character] for character in input_sequence[i]]
        target_sequence[i] = [charToInt[character] for character in target_sequence[i]]

    # Splitting data into 33 batches
    # ~1000 sentences per batch
    # batches_input = [batch_num][sentence_num]
    # Creating batch arrays
    batches_input = []
    batches_target = []
    for i in range(BATCH_SIZE):
        batches_input.append([])
        batches_target.append([])

    # FIlling the batch arrays with sentences
    for i in range(len(sentences)):
        batches_input[i%BATCH_SIZE].append(input_sequence[i])
        batches_target[i%BATCH_SIZE].append(target_sequence[i])

    # Size of training vocabulary
    vocab_size = len(charToInt)


    # Initializing the RNN model
    model = RNN(vocab_size, vocab_size, 300, 2)
    model.to(device)

    # Initializing loss function
    loss = nn.CrossEntropyLoss()

    # Initializing optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Training the RNN
    for epoch in range(10):
        # Cycling through the batches
        for batch_index in range(len(batches_input)):
            # Cycling through each input in the batch
            for input_index in range(len(batches_input[batch_index])):
                # Zeroing out gradients
                optimizer.zero_grad()
                
                # Creating a tensor for the input
                x = torch.from_numpy(create_one_hot(batches_input[batch_index][input_index], vocab_size))
                x = x.to(device)

                # Sequence output for input
                #y = torch.Tensor(batches_target[batch_index][input_index])
                y = torch.from_numpy(create_one_hot(batches_target[batch_index][input_index], vocab_size))
                y = y.to(device)

                # Running the RNN
                output, hidden = model(x)

                lossValue = loss(output, y.view(-1).long())
                lossValue.backward()
                optimizer.step()

        print("Loss: " + str(lossValue.item()))

    print(sample(model, 100, charToInt, intToChar, vocab_size))


def create_one_hot(sequence, vocab_size):
    # 1 is for batch size
    encoding = np.zeros((BATCH_SIZE, len(sequence), vocab_size), dtype=np.float32)
    for i in range(len(sequence)):
        encoding[0, i, sequence[i]] = 1
    
    return encoding


def predict(model, character, charToInt, intToChar, vocab_size):

    charInput = np.array([charToInt[c] for c in character])

    charInput = create_one_hot(charInput, vocab_size)

    charInput = torch.from_numpy(charInput)
    charInput = charInput.to(device)

    out, hidden = model(charInput)

    prob = nn.functional.softmax(out[-1], dim=0).data

    char_index = torch.max(prob, dim=0)[1].item()

    return intToChar[char_index], hidden


def sample(model, out_length, charToInt, intToChar, vocab_size, start='Q U E E N :'):
    characters = [c for c in start]
    currSize = out_length - len(characters)

    for i in range(currSize):
        character, hidden_state = predict(model, characters, charToInt, intToChar, vocab_size)
        characters.append(character)
    
    return ''.join(characters)


main()