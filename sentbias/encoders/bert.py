''' Convenience functions for handling BERT '''
import torch
import pytorch_pretrained_bert as bert


def load_model(version='bert-large-uncased'):
    ''' Load BERT model and corresponding tokenizer '''
    tokenizer = bert.BertTokenizer.from_pretrained(version)
    model = bert.BertModel.from_pretrained(version)
    model.eval()

    return model, tokenizer


def encode(model, tokenizer, texts):
    ''' Use tokenizer and model to encode texts '''
    encs = {}
    for text in texts:
        tokenized = tokenizer.tokenize(text)
        indexed = tokenizer.convert_tokens_to_ids(tokenized)
        segment_idxs = [0] * len(tokenized)
        tokens_tensor = torch.tensor([indexed])
        segments_tensor = torch.tensor([segment_idxs])
        enc, _ = model(tokens_tensor, segments_tensor, output_all_encoded_layers=False)

        enc = enc[:, 0, :]  # extract the last rep of the first input
        encs[text] = enc.detach().view(-1).numpy()
    return encs
