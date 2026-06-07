import random
from torch import argmax

def char_to_idx(char):
    base_index = ord(char.upper()) - 65
    return base_index*15

def update_feedback(guess, target, reward, guess_input):
    
    for index, char in enumerate(guess):
        
        ## First index means perfect match
        if guess[index] == target[index]:
            state_index = char_to_idx(char) + index*3
            guess_input[state_index] = 1
            reward += 0.2

        ## Second index means the letter exists in the word
        elif char in target:
            state_index = char_to_idx(char) + index*3 + 1 
            guess_input[state_index] = 1
            reward += 0.1
        
        ## Third index means that word does not contain this letter
        else:
            state_index = char_to_idx(char) + 2
            for j in range(5):
                guess_input[state_index + j*3] = 1
            reward -= 0.05

    return guess_input, reward

def get_action(state, model, eps):
    rand = random.random()
    if(eps > rand):
        return random.randrange(3200)
    else:
        return argmax(model(state))