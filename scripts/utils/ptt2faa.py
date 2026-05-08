from argparse import ArgumentParser
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import pandas as pd
import re

def parse_args():
    parser = ArgumentParser(description = "Convert GenBank file to PTT file")
    parser.add_argument("-i", "--input", type = str, required = True, help = "Input PTT file (required)")
    parser.add_argument("-f", "--fasta", type = str, required = True, help = "Input FASTA file (required)")
    parser.add_argument("-o", "--output", type = str, required = True, help = "Output PTT file (required)")
    return parser.parse_args()


def main():
    args = parse_args()

    genome_seq = SeqIO.read(args.fasta, format = "fasta")
    genome_seq = genome_seq.seq
    df = pd.read_table(args.input, sep = "\t", header = 0, skiprows = 2)    
    seq_list = []

    for i in range(0, df.shape[0]):
        location = df.loc[i, 'Location']
        strand = df.loc[i, 'Strand']        
        locus_tag = df.loc[i, 'Synonym']
        location_search = re.search("([0-9]+)\.\.([0-9]+)", location)
        start, end = int(location_search.group(1)), int(location_search.group(2))

        seq_record = genome_seq[(start-1):end].reverse_complement() if strand == '-' else genome_seq[(start-1):end]
        seq_record = seq_record.translate()
        seq = SeqRecord(seq_record, id = locus_tag, description = "")
        seq_list.append(seq)
    
    SeqIO.write(seq_list, args.output, "fasta")


if __name__ == "__main__":
    main()