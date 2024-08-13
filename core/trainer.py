from .trainers import *
models = {'GSTRGCT'}
trainer_dict = {}

def get_trainer(model):
    if model == 'GSTRGCT':
        return GSTRGCT_Trainer
    else:
        raise ValueError

for model in models:
    trainer_dict[model] = get_trainer(model)
