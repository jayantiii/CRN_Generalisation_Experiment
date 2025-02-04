# Copyright (c) Megvii Inc. All rights reserved.
import os
from argparse import ArgumentParser
#simplifies the process of training machine learning models by abstracting away many of the low-level details
#  (e.g., boilerplate code for training loops, validation, checkpointing, and logging) while preserving flexibility.
import pickle
#You define the model and training logic in a LightningModule instead of manually writing the loop.
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_summary import ModelSummary

from callbacks.ema import EMACallback
from utils.torch_dist import all_gather_object, synchronize

from .base_exp import BEVDepthLightningModel

#  """Command Line Interface for BEVDepth/CRN training and evaluation
    
#     Components:
#     1. ArgumentParser - CLI argument setup
#     2. Training params - batch size, epochs, strategy
#     3. Validation params - evaluation frequency, checkpointing
#     4. Distributed training setup - DDP strategy
    
#     Usage:
#     python tools/train.py -b 4 --evaluate --ckpt_path path/to/checkpoint
#     """
 # uses CRNLightningModel
def run_cli(model_class=BEVDepthLightningModel,
            exp_name='base_exp',
            use_ema=False,
            ckpt_path=None):
    
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser) #adds PyTorch standard Lightning trainer arguments
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-p',
                               '--predict',
                               dest='predict',
                               action='store_true',
                               help='predict model on testing set')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parser = BEVDepthLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(profiler='simple',
                        deterministic=False,
                        max_epochs=24,
                        strategy='ddp',
                        # strategy='ddp_find_unused_parameters_false',
                        num_sanity_val_steps=0,
                        check_val_every_n_epoch=1,
                        gradient_clip_val=5,
                        limit_val_batches=0.25,
                        log_every_n_steps=50,
                        enable_checkpointing=True,
                        precision=16,
                        default_root_dir=os.path.join('./outputs/', exp_name))
    args = parser.parse_args()

# checks if a random seed was provided in command line arguments
# pl.seed_everything() sets random seed for:
# PyTorch# NumPy# Python random# CUDA operations
    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = model_class(**vars(args)) #CRNLightningModel
    #whether to use Exponential Moving Average (EMA) during training
    if use_ema:
        train_dataloader = model.train_dataloader()
        ema_callback = EMACallback(
            len(train_dataloader.dataset) * args.max_epochs)
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[ema_callback, ModelSummary(max_depth=3)])
    else:
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[ModelSummary(max_depth=3)])

    if args.evaluate:
        trainer.test(model, ckpt_path=args.ckpt_path)

    elif args.predict:
        predict_step_outputs = trainer.predict(model, ckpt_path=args.ckpt_path)
        all_pred_results = list()
        all_img_metas = list()
        for predict_step_output in predict_step_outputs:
            for i in range(len(predict_step_output)):
                all_pred_results.append(predict_step_output[i][:3])
                all_img_metas.append(predict_step_output[i][3])
        synchronize()
        len_dataset = len(model.test_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        model.evaluator._format_bbox(all_pred_results, all_img_metas,
                                     os.path.dirname(args.ckpt_path))
        save_predictions(all_pred_results, all_img_metas)  # added this extra line
    else:
        if ckpt_path:
            trainer.fit(model, ckpt_path=ckpt_path)
        else:
            trainer.fit(model)

import os
import pickle

def save_predictions(results, metas, save_path="predictions.pkl"):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  
    # Save the predictions
    with open(save_path, 'wb') as f:
        pickle.dump({'pred_results': results, 'img_metas': metas}, f)
    
    print(f"Predictions saved to {save_path}")


