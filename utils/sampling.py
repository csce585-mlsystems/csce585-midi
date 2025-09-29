import torch
import torch.nn.functional as F

def sample_next_note(logits, strategy="greedy", k=5, p=0.9, temperature=1.0):
    # apply temperature scaling
    probs = F.softmax(logits / temperature, dim=-1)

    if strategy == "greedy":
        # pick the note with the highest probability
        return torch.argmax(probs, dim=-1)
    
    elif strategy == "top_k":
        # get the top k probabilities and their indices
        top_k_probs, top_k_indices = torch.topk(probs, k)
        # normalize the top k probabilities
        top_k_probs = top_k_probs / torch.sum(top_k_probs, dim=-1, keepdim=True)
        # sample from the top k
        next_note_idx = torch.multinomial(top_k_probs, 1)
        # get the actual token index
        return top_k_indices.gather(-1, next_note_idx)
    
    
    elif strategy == "top_p":
        # sort the probabilities and their indices
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # find the cutoff index where cumulative probability exceeds p
        cutoff_mask = cumulative_probs <= p
        # ensure we keep at least one token
        cutoff_mask[..., 0] = True
        # zero out probabilities beyond the cutoff
        sorted_probs = sorted_probs * cutoff_mask.float()
        # normalize the remaining probabilities
        sorted_probs = sorted_probs / torch.sum(sorted_probs, dim=-1, keepdim=True)
        # sample from the nucleus
        next_note_idx = torch.multinomial(sorted_probs, 1)
        # get the actual token index
        return sorted_indices.gather(-1, next_note_idx)
    
    elif strategy == "random":
        # sample from the full distribution
        return torch.multinomial(probs, 1)
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")