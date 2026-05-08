from argparse import ArgumentParser
import pandas as pd

def parse_args():
    parser = ArgumentParser(description = "Convert GenBank file to PTT file")
    parser.add_argument("-i", "--input", type = str, required = True, help = "Input GFF file (required)")
    parser.add_argument("-o", "--output", type = str, required = True, help = "Output PTT file (required)")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_table(args.input, sep = "\t", header = None, skiprows = 3)
    df[9] = ((df[4] - df[3] + 1 - 3) / 3).astype(int)
    df[10] = df[3].astype(str) + ".." + df[4].astype(str)
    df = df[[10, 6, 9]]
    df[11] = ['Protein_' + str(x) for x in range(1, df.shape[0]+1)]
    df[12] = ['-'] * df.shape[0]
    df[13] = ['Locus_' + str(x) for x in range(1, df.shape[0]+1)]
    df[14] = ['-'] * df.shape[0]
    df[15] = ['-'] * df.shape[0]
    df[16] = ['-'] * df.shape[0]
    df.columns = ['Location', 'Strand', 'Length', 'PID', 'Gene', 'Synonym', 'Code', 'COG', 'Product']

    num_protein = str(df.shape[0])
    description = "Genome sequence"

    with open(args.output, "w") as file:
        file.write(description + "\n")
        file.write(num_protein + " proteins\n")

    df.to_csv(args.output, index = False, sep = "\t", mode = 'a')
    

if __name__ == "__main__":
    main()