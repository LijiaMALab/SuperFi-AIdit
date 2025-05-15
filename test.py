import os
import torch
import transformers
import numpy as np
import random


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_data(seqs_, tokenizer_, task_):
    off_dict = {"AA": 'A',
                "CC": 'C',
                "GG": 'G',
                "TT": 'T',
                "AC": 'B',
                "AG": 'D',
                "AT": 'E',
                "A_": 'F',
                "_A": 'H',
                "CA": 'I',
                "CG": 'J',
                "CT": 'K',
                "C_": 'L',
                "_C": 'M',
                "GA": 'N',
                "GC": 'O',
                "GT": 'P',
                "G_": 'Q',
                "_G": 'R',
                "TA": 'S',
                "TC": 'U',
                "TG": 'V',
                "T_": 'W',
                "_T": 'X', }
    if seqs_['target'] is None:
        if task_ == 'off':
            pair = seqs_['grna'] + seqs_['pam']
        else:
            print(seqs_['pam'][0])
            pair = seqs_['grna'] + seqs_['pam'][0]
    else:
        pair = [off_dict[x + y] for x, y in zip(seqs_['grna'] + seqs_['pam'][0] + 'GG', seqs_['target'] + seqs_['pam'])]
        pair = ''.join(pair)
    input1 = tokenizer_(pair)
    input1 = torch.tensor([input1['input_ids']])
    input2 = tokenizer_(seqs_['up'] + seqs_['grna'] + seqs_['pam'] + seqs_['down'])
    input2 = torch.tensor([input2['input_ids']])
    return input1, input2


task = "nfs"
# "on"/"off"/"fs"/"nfs"
# "on":on-target editing
# "off":off-target
# "fs":on-target frameshift editing
# "nfs":on-target non-frameshift editing

grna = 'TGCTGCACACCGAGCGCGTCT'  # gRNA/length: 19~24
target = None  # NTS target/length: 19~24   input in off-target task
# target section needs to be precisely matched with grna. When there are insertions or deletions, "_" should be used to align the sequences.
pam = 'AGG'  # PAM
up = 'CCACTGCTGCTGCTGCGGC'  # NTS upstream/length: 40 - gRNA length
down = 'GCTGGTCTCCGGCGCCCCGC'  # NTS downstream/length: 20 length

seqs = {"grna": grna, "target": target, "pam": pam, "up": up, "down": down}

voc_size = {"on": 6, "off": 26, "fs": 6, "nfs": 6}
seed_everything(42)


tokenizer = transformers.AutoTokenizer.from_pretrained(
    os.path.join("tokenizer", task),
    padding_side="right",
    use_fast=True,
    trust_remote_code=True,
)

config = transformers.AutoConfig.from_pretrained(
    pretrained_model_name_or_path="C:\\Users\\11494\\Desktop\\aidit-superfi\\model\\CNNBERT",
    vocab_size=voc_size[task],
    trust_remote_code=True)
model = transformers.AutoModelForSequenceClassification.from_config(
    config=config,
    trust_remote_code=True
)
model.load_state_dict(torch.load(os.path.join('bin', task, 'pytorch_model.bin')))
model.eval()

input_ids, input_ids2 = generate_data(seqs, tokenizer, task)
with torch.no_grad():
    result = model(input_ids=input_ids, input_ids2=input_ids2)['logits'][0][0]
print(result)
