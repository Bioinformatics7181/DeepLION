# # # # # # # # # # # # # # # # #
# Coding: utf8                  #
# Author: Xinyang Qian          #
# Email: qianxy@stu.xjtu.edu.cn #
# # # # # # # # # # # # # # # # #

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import argparse
import os
import sys
import numpy as np


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
        description="Script to train DeepLION with training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample_dir",
        dest="sample_dir",
        type=str,
        help="The directory of training samples.",
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
        "--epoch",
        dest="epoch",
        type=int,
        help="The number of training epochs.",
        default=1000,
    )
    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        type=float,
        help="The learning rate used to train DeepLION.",
        default=0.001,
    )
    parser.add_argument(
        "--log_interval",
        dest="log_interval",
        type=int,
        help="The fixed number of intervals to print training conditions",
        default=100,
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        help="The device used to train DeepLION.",
        default="cpu",
    )
    parser.add_argument(
        "--flag_positive",
        dest="flag_positive",
        type=str,
        help="The flag in patient sample filename.",
        default="Patient",
    )
    parser.add_argument(
        "--flag_negative",
        dest="flag_negative",
        type=str,
        help="The flag in health individual sample filename.",
        default="Health",
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=str,
        help="Output model file in .pth format.",
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


def generate_input(sps, sp_lbs, feature_dict, feature_num, ins_num, max_len):
    # Generate input matrixes and label vectors for training DeepLION.
    xs, ys = [], []
    i = 0
    for sp in sps:
        xs.append([[[0] * feature_num] * max_len] * ins_num)
        ys.append(sp_lbs[i])
        j = 0
        for tcr in sp:
            tcr_seq = tcr[0]
            # Alignment.
            right_num = max_len - len(tcr_seq)
            tcr_seq += "X" * right_num
            # Generate matrix.
            tcr_matrix = []
            for aa in tcr_seq:
                tcr_matrix.append(feature_dict[aa.upper()])
            xs[i][j] = tcr_matrix
            j += 1
        i += 1
    xs = np.array(xs)
    xs = xs.swapaxes(2, 3)
    ys = np.array(ys)
    return xs, ys


if __name__ == "__main__":
    # Parse arguments.
    args = create_parser()

    # Read amino acid vectors.
    aa_vectors = get_features(args.aa_file)

    # Read training samples.
    training_data = []
    training_labels = []
    sample_dir = args.sample_dir if args.sample_dir[-1] == "/" else args.sample_dir + "/"
    for sample_file in os.listdir(sample_dir):
        # Read sample.
        training_data.append(read_tsv(sample_dir + sample_file, [0, 1], True))

        # Get sample label.
        if sample_file.find(args.flag_positive) != -1:
            training_labels.append(1)
        elif sample_file.find(args.flag_negative) != -1:
            training_labels.append(0)
        else:
            try:
                raise ValueError()
            except ValueError as e:
                print("Wrong sample filename! Please name positive samples with '{0}' and negative samples with '{1}'."
                      .format(args.flag_positive, args.flag_negative))
                sys.exit(1)

    # Generate input.
    input_batch, label_batch = generate_input(training_data, training_labels, aa_vectors,
                                              args.feature_num, args.tcr_num, args.max_length)
    input_batch, label_batch = torch.Tensor(input_batch), torch.LongTensor(label_batch)
    dataset = Data.TensorDataset(input_batch, label_batch)
    loader = Data.DataLoader(dataset, len(input_batch), True)

    # Set model.
    model = DeepLION(aa_num=args.max_length,
                     feature_num=args.feature_num,
                     filter_num=args.filter_num,
                     kernel_size=args.kernel_size,
                     ins_num=args.tcr_num,
                     drop_out=args.dropout).to(torch.device(args.device))
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training model.
    for epoch in range(args.epoch):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            if (epoch + 1) % args.log_interval == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Save model.
    torch.save(model.state_dict(), args.output)
    print("The trained model has been saved to: " + args.output)
