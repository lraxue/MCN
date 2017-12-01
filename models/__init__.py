from models.gru_net import GRUNet
from models.res_gru_net import ResidualGRUNet
from models.multi_res_gru_net import MultiResidualGRUNet
from models.multi_seres_gru_net import MultiSEResidualGRUNet
from models.multi_res2d3d_gru_net import MultiResidual2D3DGRUNet
from models.multi_seres2d3d_gru_net import MultiSEResidual2D3DGRUNet

MODELS = (GRUNet, ResidualGRUNet, MultiResidualGRUNet, MultiSEResidualGRUNet, MultiResidual2D3DGRUNet, MultiSEResidual2D3DGRUNet)


def get_models():
    '''Returns a tuple of sample models.'''
    return MODELS


def load_model(name):
    '''Creates and returns an instance of the model given its class name.
    The created model has a single placeholder node for feeding images.
    '''
    # Find the model class from its name
    all_models = get_models()
    mdict = {model.__name__: model for model in all_models}
    if name not in mdict:
        print('Invalid model index. Options are:')
        # Display a list of valid model names
        for model in all_models:
            print('\t* {}'.format(model.__name__))
        return None
    NetClass = mdict[name]

    return NetClass
