# skip_predictor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SkipPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, 2)  # Skip or not skip
        )

    def forward(self, state):
        return self.layers(state)

    def act(self, state):
        probs = F.softmax(self.forward(state), dim=-1)
        return torch.multinomial(probs, 1).item()

def train_skip_predictor(skip_predictor, optimizer, states, actions, rewards, num_epochs=5):
    for epoch in range(num_epochs):
        for state, action, reward in zip(states, actions, rewards):
            log_prob = F.log_softmax(skip_predictor(state), dim=-1)[action]
            loss = -log_prob * reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Usage example
input_dim = 10  # Adjust based on your feature dimension
hidden_dim = 64
skip_predictor = SkipPredictor(input_dim, hidden_dim)
optimizer = optim.Adam(skip_predictor.parameters())

# Assuming you have these data:
# states: list of tensors representing song, user, and contextual features
# actions: list of skip/not skip actions (0 or 1)
# rewards: list of rewards for each action

train_skip_predictor(skip_predictor, optimizer, states, actions, rewards)
