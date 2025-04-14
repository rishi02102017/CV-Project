import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])

    input_seqs = [item['input_seq'] for item in batch]
    target_seqs = [item['target_seq'] for item in batch]

    max_input_len = max(seq.size(0) for seq in input_seqs)
    max_target_len = max(seq.size(0) for seq in target_seqs)

    max_len = max(max_input_len, max_target_len)

    # ✅ Pad with ignore_index=-1 to avoid confusion with tokens
    input_seqs_padded = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    target_seqs_padded = pad_sequence(target_seqs, batch_first=True, padding_value=0)

    # Ensure both input and target are same length
    if input_seqs_padded.size(1) != target_seqs_padded.size(1):
        target_seqs_padded = torch.nn.functional.pad(
            target_seqs_padded,
            (0, input_seqs_padded.size(1) - target_seqs_padded.size(1)),
            value=0
        )

    return {
        'image': images,
        'input_seq': input_seqs_padded,
        'target_seq': target_seqs_padded
    }
