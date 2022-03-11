# # # # # # # # # # # # # # # # #
# Coding: utf8                  #
# Author: Xinyang Qian          #
# Email: qianxy@stu.xjtu.edu.cn #
# # # # # # # # # # # # # # # # #

import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import math


class DeepLION(nn.Module):
    def __init__(self, aa_num, feature_num, filter_num, kernel_size, ins_num, drop_out):
        super(DeepLION, self).__init__()
        self.aa_num = aa_num
        self.feature_num = feature_num
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.ins_num = ins_num
        self.drop_out = drop_out
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(self.kernel_size)
        ])
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(sum(self.filter_num), 1)
        self.fc_1 = nn.Linear(self.ins_num, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, 1, sum(self.filter_num))
        out = self.dropout(self.fc(out))
        out = out.reshape(-1, self.ins_num)
        out = self.dropout(self.fc_1(out))
        return out


def create_parser():
    parser = argparse.ArgumentParser(
        description="Script to predict samples using DeepLION.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample_dir",
        dest="sample_dir",
        type=str,
        help="The directory of samples for prediction.",
        required=True,
    )
    parser.add_argument(
        "--model_file",
        dest="model_file",
        type=str,
        help="The pretrained model file for prediction in .pth format.",
        required=True,
    )
    parser.add_argument(
        "--aa_file",
        dest="aa_file",
        type=str,
        help="The file recording animo acid vectors.",
        required=True,
    )
    parser.add_argument(
        "--tcr_num",
        dest="tcr_num",
        type=int,
        help="The number of TCRs in each sample.",
        default=100,
    )
    parser.add_argument(
        "--max_length",
        dest="max_length",
        type=int,
        help="The maximum of TCR length.",
        default=24,
    )
    parser.add_argument(
        "--feature_num",
        dest="feature_num",
        type=int,
        help="The number of features in animo acid vectors.",
        default=15,
    )
    parser.add_argument(
        "--kernel_size",
        dest="kernel_size",
        type=list,
        help="The size of kernels in the convolutional layer.",
        default=[2, 3, 4, 5, 6, 7],
    )
    parser.add_argument(
        "--filter_num",
        dest="filter_num",
        type=list,
        help="The number of the filters with corresponding kernel sizes.",
        default=[3, 3, 3, 2, 2, 1],
    )
    parser.add_argument(
        "--dropout",
        dest="dropout",
        type=float,
        help="The dropout rate in one-layer linear classifiers.",
        default=0.4,
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        help="The device used to make prediction.",
        default="cpu",
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=str,
        help="Output file in .tsv format.",
        required=True,
    )
    args = parser.parse_args()
    return args


def read_tsv(filename, inf_ind, skip_1st=False, file_encoding="utf8"):
    # Return n * m matrix "final_inf" (n is the num of lines, m is the length of list "inf_ind").
    extract_inf = []
    with open(filename, "r", encoding=file_encoding) as tsv_f:
        if skip_1st:
            tsv_f.readline()
        line = tsv_f.readline()
        while line:
            line_list = line.strip().split("\t")
            temp_inf = []
            for ind in inf_ind:
                temp_inf.append(line_list[ind])
            extract_inf.append(temp_inf)
            line = tsv_f.readline()
    return extract_inf


def get_features(filename, f_num=15):
    # Read amino acid feature file and get amino acid vectors.
    f_list = read_tsv(filename, list(range(16)), True)
    f_dict = {}
    left_num = 0
    right_num = 0
    if f_num > 15:
        left_num = (f_num - 15) // 2
        right_num = f_num - 15 - left_num
    for f in f_list:
        f_dict[f[0]] = [0] * left_num
        f_dict[f[0]] += [float(x) for x in f[1:]]
        f_dict[f[0]] += [0] * right_num
    f_dict["X"] = [0] * f_num
    return f_dict


def generate_input(sp, feature_dict, feature_num, ins_num, max_len):
    # Generate input matrix for prediction.
    xs = [[[[0] * feature_num] * max_len] * ins_num]
    i = 0
    for tcr in sp:
        tcr_seq = tcr[0]
        # Alignment.
        right_num = max_len - len(tcr_seq)
        tcr_seq += "X" * right_num
        # Generate matrix.
        tcr_matrix = []
        for aa in tcr_seq:
            tcr_matrix.append(feature_dict[aa.upper()])
        xs[0][i] = tcr_matrix
        i += 1
    xs = np.array(xs)
    xs = xs.swapaxes(2, 3)
    return xs


if __name__ == "__main__":
    # Parse arguments.
    args = create_parser()

    # Read amino acid vectors.
    aa_vectors = get_features(args.aa_file)

    # Load model.
    model = DeepLION(aa_num=args.max_length,
                     feature_num=args.feature_num,
                     filter_num=args.filter_num,
                     kernel_size=args.kernel_size,
                     ins_num=args.tcr_num,
                     drop_out=args.dropout).to(torch.device(args.device))
    model.load_state_dict(torch.load(args.model_file))
    model = model.eval()

    # Predict samples.
    sample_dir = args.sample_dir if args.sample_dir[-1] == "/" else args.sample_dir + "/"
    with open(args.output, "w", encoding="utf8") as output_file:
        output_file.write("Sample\tProbability\tPrediction\n")
        for sample_file in os.listdir(sample_dir):
            # Read sample.
            sample = read_tsv(sample_dir + sample_file, [0, 1], True)

            # Generate input.
            input_matrix = generate_input(sample, aa_vectors, args.feature_num, args.tcr_num, args.max_length)
            input_matrix = torch.Tensor(input_matrix).to(torch.device(args.device))

            # Make prediction.
            predict = model(input_matrix)
            prob = float(1 / (1 + math.exp(-predict[0][1] + predict[0][0])))
            pred = True if prob > 0.5 else False

            # Save result.
            output_file.write("{0}\t{1}\t{2}\n".format(sample_file, prob, pred))
    print("The prediction results have been saved to: " + args.output)
