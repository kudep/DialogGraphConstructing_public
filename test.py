import numpy as np

from dataset import DialogueDataset
from dataset.multiwoz import load_multiwoz
from dgac import dgac_one_stage, dgac_two_stage

from datasets import load_dataset

dataset = load_dataset("schema_guided_dstc8")
test = DialogueDataset.from_dataset(dataset['test'])
val = DialogueDataset.from_dataset(dataset['validation'])
train = DialogueDataset.from_dataset(dataset['train'])

# MULTIWOZ_PATH = '/home/mark/DialogGraphConstruction/multiwoz/data/MultiWOZ_2.2'
# CONVERT_EMB_PATH = '/home/mark/DialogGraphConstruction'

# test = DialogueDataset.from_miltiwoz_v22(
#     load_multiwoz('test', MULTIWOZ_PATH))
# val = DialogueDataset.from_miltiwoz_v22(
#     load_multiwoz('dev', MULTIWOZ_PATH))
# train = DialogueDataset.from_miltiwoz_v22(
#     load_multiwoz('train', MULTIWOZ_PATH,
#     order=[
#     'dialogues_001.json', 'dialogues_011.json', 'dialogues_007.json', 'dialogues_010.json', 
#     'dialogues_017.json', 'dialogues_005.json', 'dialogues_015.json', 'dialogues_012.json', 
#     'dialogues_016.json', 'dialogues_013.json', 'dialogues_004.json', 'dialogues_009.json', 
#     'dialogues_003.json', 'dialogues_006.json', 'dialogues_008.json', 'dialogues_002.json', 
#     'dialogues_014.json'
#     ])
# )


graph_one_stage = dgac_one_stage(train, verbosity=1)
_, _, _, one_stage_accs = graph_one_stage.success_rate(test, acc_ks=[3, 5, 10])

val_markup = graph_one_stage.get_dataset_markup(val) # node labels
transitions = graph_one_stage.get_transitions() # transition probabilities

print('One stage:')
print(one_stage_accs)

graph_two_stage = dgac_two_stage(train, verbosity=1)
_, _, _, two_stage_accs = graph_two_stage.success_rate(test, acc_ks=[3, 5, 10])

print('One stage:')
print(one_stage_accs)
print('Two stage')
print(two_stage_accs)

# MultiWoZ
# One stage:
# [0.50387511 0.66234056 0.85822404]
# Two stage
# [0.6058576  0.71570531 0.84444406]