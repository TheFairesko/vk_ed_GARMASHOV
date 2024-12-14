import torch

def generate_with_reward_guidance(
        main_model, main_tokenizer,
        reward_model, reward_tokenizer,
        N=16,
        device='cpu',
    ):
    """
    Generate text samples using a main model and select the best sample based on a reward model's guidance.

    This function generates multiple text samples from a main model, evaluates each sample using a reward model,
    and returns the sample with the highest reward score. The process is guided by the reward model to select
    the most desirable output.

    Parameters:
    main_model: The language model used to generate text samples.
    main_tokenizer: The tokenizer for main_model
    reward_model: The model used to compute reward scores for the generated samples.
    reward_tokenizer: The tokenizer for reward_model
    N (int, optional): The number of text samples to generate. Default is 16.
    device (str, optional): The device on which the computation should be performed. Default is 'cpu'.

    Returns:
    str: The generated text sample with the highest reward score.
    """

    if compute_reward is not None:
        generate_with_reward_guidance.compute_reward = compute_reward
 
    input_ids = torch.zeros((N, 1), dtype=torch.int64, device=device)

    outputs = main_model.generate(input_ids=input_ids)
    
    generated_samples = [main_tokenizer.decode(output)[len("tensor("):-1] for output in outputs]

    reward_scores = generate_with_reward_guidance.compute_reward(reward_model, reward_tokenizer, generated_samples)

    best_sample_index = reward_scores.argmax()
    best_sample = generated_samples[best_sample_index]

    return best_sample
