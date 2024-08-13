from models import *

models = ['GSTRGCT']

def get_model(model, args):
    if model == 'GSTRGCT':
        try:
            return GSTRGCT(args.num_nodes, args.in_len, args.out_len,
                           args.channel, args.embed_dim, args.g_lambda,
                           args.l_mu, args.d_model, args.n_heads,
                           args.num_layers, args.dropout, args.factor,
                           args.cheb_k, args.spatial_attention, args.temporal_attention, args.full_attention)
        except:
            return
    else:
        raise ValueError

def get_model_dict(args):
    model_dict = {}
    for model in models:
        model_dict[model] = get_model(model, args)
    return model_dict