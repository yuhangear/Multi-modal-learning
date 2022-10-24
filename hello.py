import torch
print("hello")
print("hello")
print("hello")

checkpoint_encode8 = torch.load("exp/asr_conformer_lr2e-3_warmup15k_amp_nondeterministic_phone_ctc/valid.acc.ave_10best.pth")