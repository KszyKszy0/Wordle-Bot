import torch
from model import WordleModel
from dataset import WordleDataset
from torch.utils.data import DataLoader
from helper_functions import update_feedback, get_action
from replay_memory import ReplayBuffer, memory

'''
======================
Hyperparameters
======================
'''
HIDDEN_SIZE = 400
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
EPSILON = 0.9
EPSILON_DECAY = 1e-5
EPSILON_MIN = 0.2
MODELS_PATH = "models/"
torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

policy_model = WordleModel(HIDDEN_SIZE)
target_model = WordleModel(HIDDEN_SIZE)
target_model.load_state_dict(policy_model.state_dict())

optimizer = torch.optim.Adam(policy_model.parameters(), LEARNING_RATE)
dataset = WordleDataset()
dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True)
replay_buffer = ReplayBuffer()

for (i, target_word) in enumerate(dataloader):
    ## Unsquezze
    target_word = target_word[0]
    guess_counter = 0

    ## Start guessing loop
    guess_input = torch.zeros(26*5*3)
    output = get_action(guess_input, policy_model, EPSILON)
    temp_result = dataset.words[output]
    reward = 1

    while(temp_result != target_word and guess_counter < 6):
        start_state = guess_input.clone()

        guess_input, reward = update_feedback(temp_result, target_word, reward, guess_input)
        output = get_action(guess_input, policy_model, EPSILON)
        temp_result = dataset.words[output]

        replay_buffer.push(memory(start_state, output, guess_input.detach().clone(), reward))

        EPSILON = EPSILON - EPSILON_DECAY if EPSILON > 0.2 else EPSILON
        reward -= 0.1

        if(len(replay_buffer) > 8000):
            samples = replay_buffer.sample()

