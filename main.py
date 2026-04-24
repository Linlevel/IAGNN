import argparse

import numpy as np
import torch
import random
import os
from load_data import Data
from train_eval import RunModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="IAGNN", help='IAGNN')
    parser.add_argument('--dataset', type=str, default="kinship",
                        help='FB15k, FB15k-237, WN18, WN18RR, YAGO3-10, NELL-995, ...')
    parser.add_argument('--get_best_results', type=bool, default=True, help='get best results or not')
    parser.add_argument('--get_complex_results', type=bool, default=False, help='get complex results or not')
    parser.add_argument('--num_to_eval', type=int, default=3, help='number to evaluate')
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device number')
    # learning parameters
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_iterations', type=int, default=1500, help='iterations number')
    parser.add_argument('--optimizer_method', type=str, default="adam", help='optimizer method')
    parser.add_argument('--decay_rate', type=float, default=1.0, help='decay rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--self_weight', type=float, default=0.5, help='self weight')

    # convolution parameters
    parser.add_argument('--ent_vec_dim', type=int, default=100, help='entity vector dimension')
    parser.add_argument('--rel_vec_dim', type=int, default=100, help='relation vector dimension')
    parser.add_argument('--input_dropout', type=float, default=0.3, help='input dropout')
    parser.add_argument('--feature_map_dropout', type=float, default=0.3, help='feature map dropout')
    parser.add_argument('--hidden_dropout', type=float, default=0.3, help='hidden dropout')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='drop_rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha')
    parser.add_argument('--filt_h', type=int, default=2, help='filter height')
    parser.add_argument('--filt_w', type=int, default=5, help='filter width')
    parser.add_argument('--in_channels', type=int, default=1, help='in channels')
    parser.add_argument('--out_channels', type=int, default=36, help='out channels')

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    print(args)
    cuda_device = args.cuda_device  # 获取用户指定的CUDA设备序号
    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
    # 通过设置随机数种子，固定每一次的训练结果
    seed = 777
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子（多块GPU）
    torch.backends.cudnn.deterministic = True

    data = Data(data_dir=data_dir, reverse=True,device=device)
    run = RunModel(data, modelname=args.model_name, optimizer_method=args.optimizer_method,
                   num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.learning_rate,
                   decay_rate=args.decay_rate, ent_vec_dim=args.ent_vec_dim, rel_vec_dim=args.rel_vec_dim,
                   input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout, drop_rate=args.drop_rate, dropout=args.dropout, alpha_encode=args.alpha,
                   feature_map_dropout=args.feature_map_dropout, in_channels=args.in_channels,
                   out_channels=args.out_channels, filt_h=args.filt_h, filt_w=args.filt_w,
                   label_smoothing=args.label_smoothing, num_to_eval=args.num_to_eval,
                   get_best_results=args.get_best_results, get_complex_results=args.get_complex_results,
                   alpha=args.self_weight, regular_method="", regular_rate=1e-6, device=device)
    run.train_and_eval()


if __name__ == '__main__':
    main()
