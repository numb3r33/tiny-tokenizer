base      = [bytes([i]) for i in range(256)]           # 0-255
special   = [b'<pad>', b'<bos>', b'<eos>']              # 256-…
itos      = base + special                              # id → token
stoi      = {tok:i for i,tok in enumerate(itos)}        # token → id
