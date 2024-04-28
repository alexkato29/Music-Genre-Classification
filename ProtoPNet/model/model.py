import torch

from pretrained_backbones.audio_backbone import AudioNet
from model.ppnet import PPNet

def construct_genetic_ppnet(num_classes:int, prototype_shape, model_path:str, prototype_distance_function = 'cosine', prototype_activation_function='log', fix_prototypes=True):
    # Load model from a file, removing the FC layers so it's just the encoder
    model = AudioNet(num_classes, include_connected_layers=False)
    weights = torch.load(model_path)

    for k in list(weights.keys()):
        if "fc" in k or "dropout" in k:
            del weights[k]
    
    model.load_state_dict(weights)

    return PPNet(features=model, 
                 img_size=(1, 1, 661500), 
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=None, 
                 num_classes=num_classes,
                 init_weights=True, 
                 prototype_distance_function=prototype_distance_function,
                 prototype_activation_function="linear", 
                 genetics_mode=True,
                 fix_prototypes=fix_prototypes
    )

def construct_ppnet(cfg): 
    if cfg.DATASET.NAME == "audio-waveforms":
        return construct_genetic_ppnet(
            num_classes=cfg.DATASET.NUM_CLASSES, 
            prototype_shape=cfg.DATASET.WAVEFORM.PROTOTYPE_SHAPE, 
            model_path=cfg.MODEL.BACKBONE, 
            prototype_distance_function=cfg.MODEL.PROTOTYPE_DISTANCE_FUNCTION,
            prototype_activation_function=cfg.MODEL.PROTOTYPE_ACTIVATION_FUNCTION,
            fix_prototypes=False
        ).to(cfg.MODEL.DEVICE)
    else: 
        raise NotImplementedError
