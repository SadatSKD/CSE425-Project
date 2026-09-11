"""Task 1: DistilBERT multi-label tag classifier.

``t = BERT_CLS(caption)``, ``y_hat = sigmoid(W t)``, trained with per-tag BCE.
This is the text-only semantic anchor; Tasks 2-3 must reach the same tags from
audio structure.
"""
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel


class BertTagger(nn.Module):
    def __init__(self, K, name="distilbert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(name)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.bert.config.hidden_size, K),
        )

    def encode(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]  # CLS token

    def forward(self, input_ids, attention_mask):
        return self.head(self.encode(input_ids, attention_mask))


def run_epoch_bert(model, loader, device, opt=None, crit=None):
    """One train (opt given) or eval pass. Returns (mean_loss, y_true, y_prob)."""
    train = opt is not None
    model.train() if train else model.eval()
    crit = crit or nn.BCEWithLogitsLoss()
    tot, ys, ps = 0.0, [], []
    for b in loader:
        ids = b.input_ids.to(device)
        am = b.attention_mask.to(device)
        y = b.y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(ids, am)
            loss = crit(logits, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        tot += loss.item() * y.size(0)
        ys.append(y.cpu().numpy())
        ps.append(torch.sigmoid(logits).detach().cpu().numpy())
    return tot / len(loader.dataset), np.concatenate(ys), np.concatenate(ps)
