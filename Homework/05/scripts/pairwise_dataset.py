from torch.utils.data import Dataset

class IMDBPairwiseDataset(Dataset):
    """ 
    A dataset of all possible pairs of chosen and rejected texts for TRL reward training format.

    This dataset is designed to facilitate the training of a reward model by providing pairs of
    texts where one is preferred (chosen) and the other is not (rejected). Each sample in the dataset
    is a dictionary containing tokenized input IDs and attention masks for both the chosen and rejected
    texts.

    Parameters:
    imdb: dataset to pairwise
    tokenizer: The tokenizer used to preprocess the texts
    accepted_label (int): The label that indicates a chosen text. Texts with this label are considered
                          preferred, while others are considered rejected.

    Methods:
    __len__(): Returns the total number of possible pairs of chosen and rejected texts.
    __getitem__(index): Returns a dictionary containing tokenized inputs for a specific pair of chosen
                        and rejected texts.
    """
    
    def __init__(self, imdb, tokenizer, accepted_label):
        super().__init__()
        self.tokenizer = tokenizer
        self.chosen_texts = [sample['text'] for sample in imdb if sample['label'] == accepted_label]
        self.rejected_texts = [sample['text'] for sample in imdb if sample['label'] != accepted_label]

        assert self.chosen_texts, f"no texts with label {accepted_label}"
        # print(f"Found {len(self.chosen_texts)} chosen and {len(self.rejected_texts)} rejected texts, {len(self)} pairs")

        self.column_names = [
            'input_ids_chosen', 'attention_mask_chosen',
            'input_ids_rejected', 'attention_mask_rejected'
        ]

    def __len__(self):
        return len(self.chosen_texts) * len(self.rejected_texts)

    def __getitem__(self, index: int):
        num_rejected = len(self.rejected_texts)
        chosen_idx = index // num_rejected
        rejected_idx = index % num_rejected

        chosen_text = self.chosen_texts[chosen_idx]
        rejected_text = self.rejected_texts[rejected_idx]

        chosen_tokenized = self.tokenizer(chosen_text, return_attention_mask=True, truncation=True)
        rejected_tokenized = self.tokenizer(rejected_text, return_attention_mask=True, truncation=True)

        return dict(
            input_ids_chosen=chosen_tokenized['input_ids'],
            attention_mask_chosen=chosen_tokenized['attention_mask'],
            input_ids_rejected=rejected_tokenized['input_ids'],
            attention_mask_rejected=rejected_tokenized['attention_mask'],
        )