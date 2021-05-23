''' Convenience functions for handling BERT '''
import torch
from transformers import AutoTokenizer, AutoModel


def load_model(version='bert-large-uncased'):
    ''' Load BERT model and corresponding tokenizer '''
    tokenizer = AutoTokenizer.from_pretrained(version)
    model = AutoModel.from_pretrained(version)
    model.eval()

    return model, tokenizer


def encode(model, tokenizer, texts):
    ''' Use tokenizer and model to encode texts '''
    encs = {}
    for text in texts:
        indexed = tokenizer.encode(text)
        segment_idxs = [0] * len(indexed)
        tokens_tensor = torch.tensor([indexed])
        segments_tensor = torch.tensor([segment_idxs])
        enc, _ = model(tokens_tensor, token_type_ids=segments_tensor)

        enc = enc[:, 0, :]  # extract the last rep of the first input
        encs[text] = enc.detach().view(-1).numpy()
    return encs
