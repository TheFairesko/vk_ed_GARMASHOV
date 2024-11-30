def create_prompt(sample: dict) -> str:
    """
    Generates a prompt for a multiple choice question based on the given sample.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.

    Returns:
        str: A formatted string prompt for the multiple choice question.
    """
    choices = sample["choices"]
    letters = ["A", "B", "C", "D"]

    result = ("The following are multiple choice questions (with answers) about " + sample["subject"] + ".\n" 
            + sample["question"] + "\n"
            )
    for i in range(len(choices)):
        result += letters[i] + ". " + choices[i] + "\n"

    result += "Answer:"

    return result


def create_prompt_with_examples(sample: dict, examples: list, add_full_example: bool = False) -> str:
    """
    Generates a 5-shot prompt for a multiple choice question based on the given sample and examples.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.
        examples (list): A list of 5 example dictionaries from the dev set.
        add_full_example (bool): whether to add the full text of an answer option

    Returns:
        str: A formatted string prompt for the multiple choice question with 5 examples.
    """
    result = ""
    for i in range(len(examples)):
        example_sample = examples[i]
        choices = example_sample["choices"]
        letters = ["A", "B", "C", "D"]

        result += ("The following are multiple choice questions (with answers) about " + example_sample["subject"] + ".\n" 
                + example_sample["question"] + "\n"
                )

        for j in range(len(choices)):
            result += letters[j] + ". " + choices[j] + "\n"

        result += "Answer: "
        index = example_sample["answer"]

        if add_full_example:
            result += letters[index] + ". " + choices[index]
        else:
            result += letters[index]

        result+= "\n\n"

    choices = sample["choices"]
    letters = ["A", "B", "C", "D"]

    result += ("The following are multiple choice questions (with answers) about " + sample["subject"] + ".\n" 
            + sample["question"] + "\n"
            )

    for i in range(len(choices)):
        result+= letters[i] + ". " + choices[i] + "\n"

    result += "Answer:"

    return result