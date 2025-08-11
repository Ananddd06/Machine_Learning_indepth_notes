

import torch
import torch.nn as nn
import torch.nn.functional as F

class LowLevelModule(nn.Module):
    """
    The Low-Level Module (LLM).

    This module processes detailed, step-by-step data. For a task like Sudoku,
    this could be processing a single cell or a small region of the board.
    It's designed to be fast and focused on immediate context.
    """
    def __init__(self, input_dim, hidden_dim):
        super(LowLevelModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        print(f"Initialized LowLevelModule: input_dim={input_dim}, hidden_dim={hidden_dim}")

    def forward(self, x, high_level_guidance):
        """
        Processes the input `x` with guidance from the high-level module.
        """
        # Combine detailed input with high-level context
        combined_input = x + high_level_guidance
        
        out = F.relu(self.fc1(combined_input))
        out = self.fc2(out)
        return out

class HighLevelModule(nn.Module):
    """
    The High-Level Module (HLM).

    This module operates on a more abstract, compressed representation of the
    problem space. It maintains a "memory" or "plan" over longer timescales.
    For Sudoku, this could be tracking the overall state of the board and forming a strategy.
    It uses a Recurrent Neural Network (RNN) to maintain its state.
    """
    def __init__(self, input_dim, hidden_dim):
        super(HighLevelModule, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim
        print(f"Initialized HighLevelModule: input_dim={input_dim}, hidden_dim={hidden_dim}")

    def forward(self, summary_input, hidden_state):
        """
        Updates its plan based on a summary of the low-level module's work.
        """
        output, new_hidden_state = self.rnn(summary_input, hidden_state)
        return output, new_hidden_state

class HierarchicalReasoningModel(nn.Module):
    """
    The main Hierarchical Reasoning Model (HRM).

    This model orchestrates the interaction between the high-level and low-level modules.
    The core idea is a loop where:
    1. The HLM provides guidance.
    2. The LLM acts on that guidance and the detailed input.
    3. The result of the LLM's action is summarized and fed back to the HLM.
    """
    def __init__(self, detail_dim, abstract_dim, hidden_dim, output_dim, reasoning_steps=5):
        super(HierarchicalReasoningModel, self).__init__()
        self.reasoning_steps = reasoning_steps
        
        # The dimension of the abstract summary vector
        self.abstract_dim = abstract_dim

        # The dimension of the guidance vector from HLM to LLM
        self.hidden_dim = hidden_dim

        self.low_level_module = LowLevelModule(input_dim=detail_dim, hidden_dim=hidden_dim)
        self.high_level_module = HighLevelModule(input_dim=abstract_dim, hidden_dim=hidden_dim)
        
        # A layer to create a "summary" of the low-level output for the high-level module
        self.summarizer = nn.Linear(hidden_dim, abstract_dim)
        
        # Final output layer
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        print(f"Initialized HRM: reasoning_steps={reasoning_steps}")

    def forward(self, detailed_input):
        """
        `detailed_input` represents the full, detailed problem state.
        For a 9x9 Sudoku, this could be a tensor of shape (batch_size, 81, 10) for one-hot encoded digits.
        """
        batch_size = detailed_input.size(0)
        
        # Initialize the hidden state for the high-level module (the "plan")
        high_level_hidden = torch.zeros(1, batch_size, self.hidden_dim)

        # --- The Reasoning Loop ---
        for step in range(self.reasoning_steps):
            print(f"Reasoning Step {step + 1}/{self.reasoning_steps}")
            
            # 1. High-level module provides guidance from its current hidden state.
            # We use the output of the GRU which is shaped (batch_size, 1, hidden_dim)
            # and squeeze it to match the detailed input steps.
            high_level_guidance = high_level_hidden.permute(1, 0, 2) # (batch_size, 1, hidden_dim)

            # 2. Low-level module processes each detailed part of the input.
            # In a real scenario, you might iterate through parts of the input (e.g., Sudoku cells).
            # For simplicity, we process the whole detailed input at once, broadcasting the guidance.
            low_level_output = self.low_level_module(detailed_input, high_level_guidance)
            
            # 3. Summarize the low-level output to feed back to the high-level module.
            # We take the mean of the outputs across the "detail" dimension.
            summary = torch.mean(low_level_output, dim=1, keepdim=True) # (batch_size, 1, hidden_dim)
            abstract_summary = self.summarizer(summary) # (batch_size, 1, abstract_dim)
            
            # 4. Update the high-level module's state with the new summary.
            _, high_level_hidden = self.high_level_module(abstract_summary, high_level_hidden)

        # After all reasoning steps, use the final state to produce an output
        final_output = self.output_projection(high_level_hidden.squeeze(0))
        return final_output

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Setting up a dummy problem for the HRM ---")
    
    # Parameters for a hypothetical problem
    BATCH_SIZE = 1  # How many problems to solve at once
    DETAIL_DIM = 10 # Dimension of a single piece of data (e.g., one-hot-encoded digit 0-9)
    NUM_DETAILS = 81 # Number of details (e.g., 81 cells in a Sudoku grid)
    ABSTRACT_DIM = 32 # The size of the compressed "summary" vector
    HIDDEN_DIM = 64   # The size of the internal state and guidance vectors
    OUTPUT_DIM = 810  # Final output dimension (e.g., 81 cells * 10 possible digits)
    REASONING_STEPS = 4 # How many cycles of high-level/low-level interaction

    # Create the model
    model = HierarchicalReasoningModel(
        detail_dim=DETAIL_DIM,
        abstract_dim=ABSTRACT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        reasoning_steps=REASONING_STEPS
    )
    
    # Create some dummy input data
    # This represents a batch of Sudoku-like problems
    dummy_input = torch.randn(BATCH_SIZE, NUM_DETAILS, DETAIL_DIM)
    print(f"\n--- Running forward pass with dummy data of shape: {dummy_input.shape} ---")

    # Run the model
    output = model(dummy_input)
    
    print(f"\n--- Forward pass complete ---")
    print(f"Final output shape: {output.shape}")
    print(f"Expected output shape: ({BATCH_SIZE}, {OUTPUT_DIM})")

