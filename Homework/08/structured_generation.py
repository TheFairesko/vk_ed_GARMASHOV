import random

from fsm import FSM, build_odd_zeros_fsm


def get_valid_tokens(vocab: dict[int, str], eos_token_id: int, fsm: FSM, state: int) -> list[int]:
    """Filter tokens from the vocabulary based on the given state in the FSM.  
    1. Retain only tokens that can be achieved from the given state.  
    2. If the current state is terminal, then add the EOS token.

    Args:
        vocab (dict): vocabulary, id to token
        eos_token_id (int): index of EOS token
        fsm (FSM): Finite-State Machine
        state (int): start state
    Returns:
        valid tokens (list): list of possible tokens
    """
    valid_tokens = []

    for token_id in vocab.keys():
        if token_id == eos_token_id and fsm.is_terminal(state):
            valid_tokens.append(token_id)
        elif fsm.validate_continuation(state, vocab[token_id]):
            valid_tokens.append(token_id)

    return valid_tokens


def random_generation() -> str:
    """Structured generation based on Odd-Zeros FSM with random sampling from possible tokens.

    Args:
    Returns:
        generation (str): A binary string with an odd number of zeros.
    """
    # Define our vocabulary
    vocab = {0: "[EOS]", 1: "0", 2: "1"}
    eos_token_id = 0
    # Init Finite-State Machine
    fsm, state = build_odd_zeros_fsm()

    # List with generate tokens
    tokens: list[int] = []
    # Sample until EOS token
    while True:
        # 1. Get valid tokens
        valid_tokens = get_valid_tokens(vocab, eos_token_id, fsm, state)
        # 2. Get next token
        next_token = random.choice(valid_tokens)

        # 3. End generation or move to next iteration
        if next_token == eos_token_id:
            break
        else:
            tokens.append(next_token)
            state = fsm.states.index(fsm.states[state].transitions[vocab[next_token]])

    # Convert tokens to string
    return "".join([vocab[it] for it in tokens])


if __name__ == "__main__":
    print(random_generation())
