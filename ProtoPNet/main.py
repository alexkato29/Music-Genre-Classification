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

import matplotlib.pyplot as plt


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
    log, logclose = create_logger(log_filename=os.path.join(cfg.OUTPUT.MODEL_DIR, '427_continued_training.log'))
    log(str(cfg))

    # Print GPUs (multiple GPU clusters at once?)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs being used: {num_gpus}", flush=True)
    else:
        print("CUDA is not available. No GPUs are being used.", flush=True)
    
    # Get the dataset for training
    train_loader, push_loader, val_loader, _ = get_dataset(cfg)

    # Construct and parallel the model
    # ppnet = construct_ppnet(cfg)
    target_acc = 0.7
    ppnet = torch.load(cfg.OUTPUT.MODEL_DIR + "/427backbone_5_push0.6107.pth")  # This continues training the existing model
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

    # Save for a graph
    train_accs = []
    val_accs = []

    for epoch in range(cfg.OPTIM.NUM_TRAIN_EPOCHS):
        log('epoch: \t{0}'.format(epoch))
        
        # Warm up and Training Epochs
        if epoch < cfg.OPTIM.NUM_WARM_EPOCHS:
            tnt.warm_only(model=ppnet_multi, log=log)
            accu = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log)
            train_accs.append(accu)
        else:
            tnt.joint(model=ppnet_multi, log=log)
            joint_lr_scheduler.step()
            accu = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log)
            train_accs.append(accu)

        # Validation Epochs
        accu = tnt.test(model=ppnet_multi, dataloader=val_loader,
                        class_specific=class_specific, log=log)
        val_accs.append(accu)

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

            # Optimize last layer
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(cfg.OPTIM.NUM_PUSH_EPOCHS):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                            class_specific=class_specific, coefs=coefs, log=log)
                accu = tnt.test(model=ppnet_multi, dataloader=val_loader,
                                class_specific=class_specific, log=log)
                save_model_w_condition(model=ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + '_push_427_', accu=accu, 
                                       target_accu=target_acc, log=log)

    logclose()

    plt.figure(figsize=(12, 6))
    plt.plot(train_accs, label='Training Acc')
    plt.plot(val_accs, label='Validation Acc')
    plt.title('Training and Validation Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training.png')
        


if __name__ == '__main__':
    main()
    
