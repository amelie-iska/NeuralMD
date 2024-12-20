import argparse
import os
from tqdm import tqdm
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchdiffeq import odeint, odeint_adjoint
from torch_ema import ExponentialMovingAverage

from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleMultiTrajectory
from NeuralMD.dataloaders.dataloader_MISATO import DataLoaderMISATO
from NeuralMD.evaluation import get_matching_list, get_stability_list, get_ligand_collision_list, get_binding_collision_list_semi_flexible
from models.NeuralMD_Binding01_2nd_ODE import NeuralMD_Binding01
from models.NeuralMD_Binding02_2nd_SDE import NeuralMD_Binding02
from torch_ema import ExponentialMovingAverage


def save_model(save_best):
    if not args.output_model_dir == "":
        if save_best:
            print("save model with optimal loss")
            output_model_path = os.path.join(args.output_model_dir, "model.pth")
            saved_model_dict = {}
            saved_model_dict["binding_model"] = binding_model.state_dict()
            torch.save(saved_model_dict, output_model_path)

        else:
            print("save model in the last epoch")
            output_model_path = os.path.join(args.output_model_dir, "model_final.pth")
            saved_model_dict = {}
            saved_model_dict["binding_model"] = binding_model.state_dict()
            torch.save(saved_model_dict, output_model_path)
    return


def train(loader):
    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader

    accum_loss, accum_loss_pos, accum_loss_velocity, accum_count = 0, 0, 0, 0
    start_time = time.time()

    for batch in L:
        batch = batch.to(device)

        end_traj_idx = random.randint(1, 99)
        if args.NeuralMD_Binding_start_with_first_frame:
            start_traj_idx = 0
        else:
            start_traj_idx = max(0, end_traj_idx - args.NeuralMD_Binding_frame_num)

        step_size = args.NeuralMD_step_size / args.NeuralMD_scaling
        traj_idx_tensor = torch.tensor(np.arange(start_traj_idx, end_traj_idx+1), dtype=torch.float32).to(device) / args.NeuralMD_scaling
        traj_idx_tensor = traj_idx_tensor - traj_idx_tensor[0]

        ligand_pos_init = batch.ligand_trajectory_pos[:, start_traj_idx, :].clone()
        velocity_init = batch.ligand_trajectory_pos[:, start_traj_idx+1, :] - batch.ligand_trajectory_pos[:, start_traj_idx, :]

        condition_ligand = (batch.ligand_x, batch.batch_ligand, batch.ligand_mass)
        pos_N = batch.protein_pos[batch.mask_n]
        pos_Ca = batch.protein_pos[batch.mask_ca]
        pos_C = batch.protein_pos[batch.mask_c]
        condition_protein = (pos_N, pos_Ca, pos_C, batch.protein_backbone_residue, batch.batch_residue)
        condition = condition_ligand + condition_protein

        if args.use_MLP_velocity:
            _, velocity_init = binding_model.velocity_model(
                z=condition[0],
                pos=velocity_init,
                batch=condition[1])
        output = odeint(binding_model, (velocity_init, ligand_pos_init), traj_idx_tensor, condition=condition, method=args.ODE_method, options={"step_size": step_size})
        output_velocity_pred, output_pos_pred = output
        assert output_pos_pred.shape[0] == end_traj_idx - start_traj_idx + 1
        
        loss_pos, loss_velocity = 0, 0
        for traj_idx in range(start_traj_idx+1, end_traj_idx+1):
            ligand_pos_target = batch.ligand_trajectory_pos[:, traj_idx, :].clone()
            ligand_pos_pred = output_pos_pred[traj_idx - start_traj_idx]
            loss_pos = loss_pos + criterion(ligand_pos_target, ligand_pos_pred)

        loss_pos = loss_pos / (end_traj_idx - start_traj_idx)

        loss = args.loss_pos_coefficient * loss_pos
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ema.update()

        accum_loss_pos += loss_pos.item()
        accum_loss += loss.item()
        accum_count += 1
    
    if args.lr_scheduler in ["StepLR", "CosineAnnealingLR"]:
        lr_scheduler.step()
    elif args.lr_scheduler in ["ReduceLROnPlateau"]:
        lr_scheduler.step(accum_loss / accum_count)

    print("loss pos: {:.5f}\tloss velocity: {:.5f}\t{:.3f}s".format(
        accum_loss_pos / accum_count, accum_loss_velocity/ accum_count, time.time() - start_time))
    return


@torch.no_grad()
def evaluate(loader):
    start_time = time.time()
    binding_model.eval()

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader

    mse_loss, mae_loss, eval_count = 0, 0, 0
    matching_list, stability_list, ligand_collision_list, binding_collision_list = [], [], [], []

    for batch in L:
        batch = batch.to(device)

        start_traj_idx, end_traj_idx = 0, 99

        ligand_pos_init = batch.ligand_trajectory_pos[:, start_traj_idx, :]
        velocity_init = batch.ligand_trajectory_pos[:, start_traj_idx+1, :] - batch.ligand_trajectory_pos[:, start_traj_idx, :]
        
        step_size = args.NeuralMD_step_size / args.NeuralMD_scaling
        traj_idx_tensor = torch.tensor(np.arange(start_traj_idx, end_traj_idx+1), dtype=torch.float32).to(device) / args.NeuralMD_scaling
        traj_idx_tensor = traj_idx_tensor - traj_idx_tensor[0]

        condition_ligand = (batch.ligand_x, batch.batch_ligand, batch.ligand_mass)
        pos_N = batch.protein_pos[batch.mask_n]
        pos_Ca = batch.protein_pos[batch.mask_ca]
        pos_C = batch.protein_pos[batch.mask_c]
        condition_protein = (pos_N, pos_Ca, pos_C, batch.protein_backbone_residue, batch.batch_residue)
        condition = condition_ligand + condition_protein

        # output: tuple(velocity, pos)
        if args.use_MLP_velocity:
            _, velocity_init = binding_model.velocity_model(
                z=condition[0],
                pos=velocity_init,
                batch=condition[1])
        output = odeint(binding_model, (velocity_init, ligand_pos_init), traj_idx_tensor, condition=condition, method=args.ODE_method, options={"step_size": step_size})
        output_velocity_pred, output_pos_pred = output
        assert output_pos_pred.shape[0] == end_traj_idx - start_traj_idx + 1

        for traj_idx in range(start_traj_idx+1, end_traj_idx+1):
            pos_target = batch.ligand_trajectory_pos[:, traj_idx, :]
            pos_pred = output_pos_pred[traj_idx - start_traj_idx]
            
            mse_loss = mse_loss + torch.sum((pos_pred - pos_target).pow(2).sum(dim=1).sqrt()).cpu().item()
            mae_loss = mae_loss + torch.sum(torch.abs(pos_pred - pos_target)).cpu().item()

            eval_count += pos_pred.shape[0]
    
        trajectory_target = batch.ligand_trajectory_pos.cpu()[:, 1:, :].transpose(0, 1)
        trajectory_pred = output_pos_pred.cpu()[1:, :, :]
        matching_list.extend(get_matching_list(trajectory_target, trajectory_pred, batch=batch.batch_ligand))
        stability_list.extend(get_stability_list(trajectory_target, trajectory_pred, batch=batch.batch_ligand))
        ligand_collision_list.extend(get_ligand_collision_list(trajectory_pred, batch.ligand_x, batch=batch.batch_ligand))
        batch_protein = batch.batch_residue.unsqueeze(0).expand([3, -1]).contiguous().view(-1)
        protein_x = torch.ones((batch_protein.shape[0],))
        protein_x[batch.mask_n] = 6
        protein_x[batch.mask_ca] = 5
        protein_x[batch.mask_c] = 5
        binding_collision_list.extend(get_binding_collision_list_semi_flexible(trajectory_pred, batch.ligand_x, batch.protein_pos.cpu(), protein_x, batch_ligand=batch.batch_ligand, batch_protein=batch_protein))


    mae_loss = mae_loss / eval_count
    mse_loss = mse_loss / eval_count
    matching = np.mean(matching_list)
    stability = np.mean(stability_list)
    ligand_collision = np.mean(ligand_collision_list)
    binding_collision = np.mean(binding_collision_list)
    
    total_frame_count = 99 * len(matching_list)
    total_time = time.time() - start_time
    FPS = total_frame_count / total_time
    return mae_loss, mse_loss, matching, stability, ligand_collision, binding_collision, FPS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--loss", type=str, default="MSE", choices=["MSE", "MAE"])
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "Adam", "AdamW"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="None")
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--print_every_epoch", type=int, default=5)
    parser.add_argument("--output_model_dir", type=str, default="")
    
    parser.add_argument("--input_data_dir", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="MISATO_100")

    parser.add_argument("--eval_train", dest="eval_train", action="store_true")
    parser.add_argument("--no_eval_train", dest="eval_train", action="store_false")
    parser.set_defaults(eval_train=False)

    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--no_verbose", dest="verbose", action="store_false")
    parser.set_defaults(verbose=False)

    # for NeuralMD-Binding
    parser.add_argument("--NeuralMD_Binding_start_with_first_frame", dest="NeuralMD_Binding_start_with_first_frame", action="store_true")
    parser.add_argument("--no_NeuralMD_Binding_start_with_first_frame", dest="NeuralMD_Binding_start_with_first_frame", action="store_false")
    parser.set_defaults(NeuralMD_Binding_start_with_first_frame=True)
    parser.add_argument("--NeuralMD_Binding_frame_num", type=int, default=10)
    parser.add_argument("--NeuralMD_step_size", type=float, default=1)
    parser.add_argument("--NeuralMD_scaling", type=float, default=1)
    parser.add_argument("--NeuralMD_velocity_refined_value_coefficient", type=float, default=1)
    parser.add_argument("--NeuralMD_binding_model", type=str, default="NeuralMD_Binding01")
    parser.add_argument("--ODE_method", type=str, default="euler", choices=["euler", "rk4", "euler_adjoint", "rk4_adjoint"])

    parser.add_argument("--use_MLP_velocity", dest="use_MLP_velocity", action="store_true")
    parser.add_argument("--no_MLP_velocity", dest="use_MLP_velocity", action="store_false")
    parser.set_defaults(use_MLP_velocity=False)

    parser.add_argument("--model_3d_ligand", type=str, default="FrameNet01")
    parser.add_argument("--model_3d_protein", type=str, default="FrameNetProtein03")

    parser.add_argument("--loss_pos_coefficient", type=float, default=1)

    # for modeling
    parser.add_argument("--emb_dim", type=int, default=128)

    # for FrameNet
    parser.add_argument("--FrameNet_cutoff", type=float, default=5)
    parser.add_argument("--FrameNet_num_layers", type=int, default=4)
    parser.add_argument("--FrameNet_complex_layer", type=int, default=1)
    parser.add_argument("--FrameNet_num_radial", type=int, default=96)
    parser.add_argument("--FrameNet_rbf_type", type=str, default="RBF_repredding_01")
    parser.add_argument("--FrameNet_gamma", type=float, default=None)
    parser.add_argument("--FrameNet_readout", type=str, default="mean")

    args = parser.parse_args()
    print("args", args)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if "adjoint" in args.ODE_method:
        odeint = odeint_adjoint
        args.ODE_method = args.ODE_method.replace("_adjoint", "")
    else:
        odeint = odeint

    data_root = os.path.join(args.input_data_dir, args.dataset)
    train_dataset = DatasetMISATOSemiFlexibleMultiTrajectory(data_root, mode="train")
    val_dataset = DatasetMISATOSemiFlexibleMultiTrajectory(data_root, mode="val")
    test_dataset = DatasetMISATOSemiFlexibleMultiTrajectory(data_root, mode="test")
    print("len of train", len(train_dataset))
    print("len of val", len(val_dataset))
    print("len of test", len(test_dataset))

    train_loader = DataLoaderMISATO(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoaderMISATO(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_loader = DataLoaderMISATO(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    if args.NeuralMD_binding_model == "NeuralMD_Binding01":
        binding_model = NeuralMD_Binding01(args).to(device)
    elif args.NeuralMD_binding_model == "NeuralMD_Binding02":
        binding_model = NeuralMD_Binding02(args).to(device)

    if args.loss == "MSE":
        criterion = nn.MSELoss(reduction="mean")
    elif args.loss == "MAE":
        criterion = nn.L1Loss(reduction="mean")

    # set up optimizer
    model_param_group = [
        {"params": binding_model.parameters(), "lr": args.lr},
    ]
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model_param_group, lr=args.lr, weight_decay=args.decay)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model_param_group, lr=args.lr, weight_decay=args.decay)
    ema = ExponentialMovingAverage(binding_model.parameters(), decay=0.995)

    lr_scheduler = None
    if args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs
        )
        print("Apply lr scheduler CosineAnnealingLR")
    elif args.lr_scheduler == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor
        )
        print("Apply lr scheduler StepLR")
    elif args.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.lr_decay_factor, patience=args.lr_decay_patience, min_lr=args.min_lr
        )
        print("Apply lr scheduler ReduceLROnPlateau")

    train_mae_list, val_mae_list, test_mae_list = [], [], []
    train_mse_list, val_mse_list, test_mse_list = [], [], []
    train_matching_list, val_matching_list, test_matching_list = [], [], []
    train_stability_list, val_stability_list, test_stability_list = [], [], []
    train_ligand_collision_list, val_ligand_collision_list, test_ligand_collision_list = [], [], []
    train_binding_collision_list, val_binding_collision_list, test_binding_collision_list = [], [], []
    best_val_mae, best_val_idx = 1e10, 0

    print("Initial")
    val_mae, val_mse, val_matching, val_stability, val_ligand_collision, val_binding_collision, _ = evaluate(val_loader)
    test_mae, test_mse, test_matching, test_stability, test_ligand_collision, test_binding_collision, FPS = evaluate(test_loader)
    print("MAE val: {:.5f}\t\ttest: {:.5f}".format(val_mae, test_mae))
    print("MSE val: {:.5f}\t\ttest: {:.5f}".format(val_mse, test_mse))
    print("hr MAE val: {:.5f}\t\ttest: {:.5f}".format(val_matching, test_matching))
    print("Stability val: {:.5f}\t\ttest: {:.5f}".format(val_stability, test_stability))
    print("Ligand collision val: {:.5f}\t\ttest: {:.5f}".format(val_ligand_collision, test_ligand_collision))
    print("Binding collision val: {:.5f}\t\ttest: {:.5f}".format(val_binding_collision, test_binding_collision))
    print("FPS: {:.5f}".format(FPS))
    print()

    for e in range(1, args.epochs+1):
        print("epoch {}".format(e))
        train(train_loader)

        if e % args.print_every_epoch == 0:
            if args.eval_train:
                train_mae, train_mse, train_matching, train_stability, train_ligand_collision, train_binding_collision, _ = evaluate(train_loader)
            else:
                train_mae, train_mse, train_matching, train_stability, train_ligand_collision, train_binding_collision = 0, 0, 0, 0, 0, 0
            val_mae, val_mse, val_matching, val_stability, val_ligand_collision, val_binding_collision, _ = evaluate(val_loader)
            test_mae, test_mse, test_matching, test_stability, test_ligand_collision, test_binding_collision, _ = evaluate(test_loader)

            train_mae_list.append(train_mae)
            train_mse_list.append(train_mse)
            train_matching_list.append(train_matching)
            train_stability_list.append(train_stability)
            train_ligand_collision_list.append(train_ligand_collision)
            train_binding_collision_list.append(train_binding_collision)
            val_mae_list.append(val_mae)
            val_mse_list.append(val_mse)
            val_matching_list.append(val_matching)
            val_stability_list.append(val_stability)
            val_ligand_collision_list.append(val_ligand_collision)
            val_binding_collision_list.append(val_binding_collision)
            test_mae_list.append(test_mae)
            test_mse_list.append(test_mse)
            test_matching_list.append(test_matching)
            test_stability_list.append(test_stability)
            test_ligand_collision_list.append(test_ligand_collision)
            test_binding_collision_list.append(test_binding_collision)

            if val_mae <= best_val_mae:
                best_val_mae = val_mae
                best_val_idx = len(train_mae_list) - 1
                save_model(save_best=True)

            print("MAE train: {:.5f}\t\tval: {:.5f}\t\ttest: {:.5f}".format(train_mae, val_mae, test_mae))
            print("MSE train: {:.5f}\t\tval: {:.5f}\t\ttest: {:.5f}".format(train_mse, val_mse, test_mse))
            print("hr MAE train: {:.5f}\t\tval: {:.5f}\t\ttest: {:.5f}".format(train_matching, val_matching, test_matching))
            print("Stability train:{:.5f}\t\tval: {:.5f}\t\ttest: {:.5f}".format(train_stability, val_stability, test_stability))
            print("Ligand collision train:{:.5f}\t\tval: {:.5f}\t\ttest: {:.5f}".format(train_ligand_collision, val_ligand_collision, test_ligand_collision))
            print("Binding collision train:{:.5f}\t\tval: {:.5f}\t\ttest: {:.5f}".format(train_binding_collision, val_binding_collision, test_binding_collision))
            print()
    
    print("best MAE train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
        train_mae_list[best_val_idx], val_mae_list[best_val_idx], test_mae_list[best_val_idx],
    ))
    print("best MSE train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
        train_mse_list[best_val_idx], val_mse_list[best_val_idx], test_mse_list[best_val_idx],
    ))
    print("best hr MAE train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
        train_matching_list[best_val_idx], val_matching_list[best_val_idx], test_matching_list[best_val_idx],
    ))
    print("best Stability train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
        train_stability_list[best_val_idx], val_stability_list[best_val_idx], test_stability_list[best_val_idx],
    ))
    print("best Ligand collision train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
        train_ligand_collision_list[best_val_idx], val_ligand_collision_list[best_val_idx], test_ligand_collision_list[best_val_idx],
    ))
    print("best Binding collision train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
        train_binding_collision_list[best_val_idx], val_binding_collision_list[best_val_idx], test_binding_collision_list[best_val_idx],
    ))
    print("FPS: {:.5f}".format(FPS))
    save_model(save_best=False)