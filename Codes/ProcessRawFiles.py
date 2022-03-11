# # # # # # # # # # # # # # # # #
# Coding: utf8                  #
# Author: Xinyang Qian          #
# Email: qianxy@stu.xjtu.edu.cn #
# # # # # # # # # # # # # # # # #

import argparse


def create_parser():
    parser = argparse.ArgumentParser(
        description="Script to process raw files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        dest="input",
        type=str,
        help="The input raw file in .tsv format.",
        required=True,
    )
    parser.add_argument(
        "--reference",
        dest="reference",
        type=str,
        help="The reference dataset in .tsv format.",
        default="",
    )
    parser.add_argument(
        "--tcr_num",
        dest="tcr_num",
        type=int,
        help="The number of TCRs extracted from the raw file after filteration process.",
        default=100,
    )
    parser.add_argument(
        "--info_index",
        dest="info_index",
        type=list,
        help="The index list of the used information in the raw file.",
        default=[-3, 2],
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


def filter_sequence(raw, reference):
    # Filtering low-quality sequences and the ones in the reference dataset.
    aa_list = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    result = []
    save_flag = 1
    for seq in raw:
        if len(seq[0]) > 24 or len(seq[0]) < 10:
            save_flag = 0
        for aa in seq[0]:
            if aa not in aa_list:
                save_flag = 0
        if seq[0][0].upper() != "C" or seq[0][-1].upper() != "F":
            save_flag = 0
        if [seq[0]] in reference:
            save_flag = 0
        if save_flag == 1:
            result.append(seq)
        else:
            save_flag = 1
    return result


if __name__ == "__main__":
    # Parse arguments.
    args = create_parser()

    # Read the raw file.
    raw_file = read_tsv(args.input, args.info_index, True)

    # Read the reference dataset.
    if args.reference != "":
        reference_file = read_tsv(args.reference, [0])
    else:
        reference_file = []

    # Extract TCRs.
    processed_file = filter_sequence(raw_file, reference_file)
    output_file = sorted(processed_file, key=lambda x: float(x[1]), reverse=True)[: 100]

    # Save output.
    with open(args.output, "w", encoding="utf8") as output_f:
        output_f.write("TCR\tAbundance\n")
        for tcr in output_file:
            output_f.write("{0}\t{1}\n".format(tcr[0], tcr[1]))
    print("The processed file has been saved to: " + args.output)
