import numpy as np
import torch


class MlmDataset(torch.utils.data.Dataset):
    def __init__(self, rxn, tokenizer, max_len=400):
        self.tokenizer = tokenizer
        self.rxn = rxn.values.tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.rxn)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(self.rxn[index],
                                            None,
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            pad_to_max_length=True,
                                            return_token_type_ids=True)

        return {'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
                'token_type_ids': torch.tensor(inputs["token_type_ids"], dtype=torch.long),
                'labels': torch.tensor(inputs['input_ids'], dtype=torch.long)}


class RegDataset(torch.utils.data.Dataset):
    def __init__(self, rxn, targets, tokenizer, max_len=200):
        self.tokenizer = tokenizer
        self.rxn = rxn.values.tolist()
        self.max_len = max_len
        self.targets = np.array(targets.values.tolist()).tolist()

    def __len__(self):
        return len(self.rxn)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(self.rxn[index],
                                            None,
                                            add_special_tokens=True,
                                            padding='max_length',
                                            max_length=self.max_len,
                                            return_token_type_ids=True)

        return {'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
                'token_type_ids': torch.tensor(inputs["token_type_ids"], dtype=torch.long),
                'labels': torch.tensor(self.targets[index], dtype=torch.float)}
