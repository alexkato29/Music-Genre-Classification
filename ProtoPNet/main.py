import argparse, os
import torch
from utils.util import save_model_w_condition, create_logger
from os import mkdir

from  configs.cfg import get_cfg_defaults
from dataio.dataset import get_dataset

from model.model import construct_ppnet
from model.utils import get_optimizers

import train.train_and_test as tnt

import prototype.push as push       


def main():
    cfg = get_cfg_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default='0') 
    parser.add_argument('--configs', type=str, default='cub.yaml')
    args = parser.parse_args()

    # Update the hyperparameters from default to the ones we mentioned in arguments
    cfg.merge_from_file(args.configs)
    
    if not os.path.exists(cfg.OUTPUT.MODEL_DIR):
        mkdir(cfg.OUTPUT.MODEL_DIR)
    if not os.path.exists(cfg.OUTPUT.IMG_DIR):
        mkdir(cfg.OUTPUT.IMG_DIR)

    # Create Logger Initially
    log, logclose = create_logger(log_filename=os.path.join(cfg.OUTPUT.MODEL_DIR, 'train.log'))
    # log, logclose = create_logger(log_filename=os.path.join("stuff", 'train.log'))
    log(str(cfg))
    
    # Get the dataset for training
    train_loader, push_loader, val_loader, _ = get_dataset(cfg)

    # Construct and parallel the model
    ppnet = construct_ppnet(cfg)
    ppnet_multi = torch.nn.DataParallel(ppnet) 
    class_specific = True
    
    joint_optimizer, joint_lr_scheduler, warm_optimizer, last_layer_optimizer = get_optimizers(cfg, ppnet)

    log('start training')
    
    # Prepare loss function
    coefs = {
        'crs_ent': cfg.OPTIM.COEFS.CRS_ENT,
        'clst': cfg.OPTIM.COEFS.CLST,
        'sep': cfg.OPTIM.COEFS.SEP,
        'l1': cfg.OPTIM.COEFS.L1
    }

    for epoch in range(cfg.OPTIM.NUM_TRAIN_EPOCHS):
        log('epoch: \t{0}'.format(epoch))
        
        # Warm up and Training Epochs
        if epoch < cfg.OPTIM.NUM_WARM_EPOCHS:
            tnt.warm_only(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log)
        else:
            tnt.joint(model=ppnet_multi, log=log)
            joint_lr_scheduler.step()
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log)

        # Testing Epochs
        accu = tnt.test(model=ppnet_multi, dataloader=val_loader,
                        class_specific=class_specific, log=log)
        save_model_w_condition(model=ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + 'nopush', accu=accu,
                                    target_accu=0.70, log=log)

    # Pushing Epochs
        print(os.path.join(cfg.OUTPUT.IMG_DIR, str(epoch) + '_' + 'push_weights.pth'))
        if epoch >= cfg.OPTIM.PUSH_START and epoch in cfg.OPTIM.PUSH_EPOCHS:
            push.push_prototypes(
                push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=cfg.OUTPUT.PREPROCESS_INPUT_FUNCTION, # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=cfg.OUTPUT.IMG_DIR, # if not None, prototypes will be saved here
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX,
                prototype_self_act_filename_prefix=cfg.OUTPUT.PROTOTYPE_SELF_ACT_FILENAME_PREFIX,
                proto_bound_boxes_filename_prefix=cfg.OUTPUT.PROTO_BOUND_BOXES_FILENAME_PREFIX,
                save_prototype_class_identity=True,
                log=log,
                no_save=cfg.OUTPUT.NO_SAVE,
                fix_prototypes=False)
            
            accu = tnt.test(model=ppnet_multi, dataloader=val_loader,
                            class_specific=class_specific, log=log)
            save_model_w_condition(model=ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + 'push', accu=accu,
                                        target_accu=0.5, log=log)

            # Optimize last layer
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(20):
            # for i in range(1):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                            class_specific=class_specific, coefs=coefs, log=log)
                accu = tnt.test(model=ppnet_multi, dataloader=val_loader,
                                class_specific=class_specific, log=log)
                save_model_w_condition(model=ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + '_' + 'push', accu=accu, target_accu=0.70, log=log)

            # Print the weights of the last layer
            # Save the weigts of the last layer
            torch.save(ppnet.last_layer.state_dict(), os.path.join(cfg.OUTPUT.IMG_DIR, str(epoch) + '_push_weights.pth'))

    logclose()
        


if __name__ == '__main__':
    main()
    