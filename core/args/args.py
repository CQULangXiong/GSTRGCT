from .gstrgct_args import GSTRGCT_GenArgs


def gen_model_args(model):
    if model == 'GSTRGCT':
        return GSTRGCT_GenArgs
    else:
        raise ValueError

gen_args = {}
models = {'GSTRGCT'}
for model in models:
    gen_args[model] = gen_model_args(model)