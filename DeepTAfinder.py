"""
Licensed under the MIT License
"""

from argparse import ArgumentParser
from unittest import result
from xml.parsers.expat import model
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import numpy as np
import os
import re
import sys
import subprocess
import shutil

def parse_args():
    parser = ArgumentParser(description = "Prediction of type II TA system.")
    parser.add_argument("-i", "--input", type = str, required = True,
                        help = "(required) Input genome sequence file")
    parser.add_argument("-m", "--model", type = str, required = True, 
                        help = "Model location")
    parser.add_argument("-f", "--format", type = str, required = True, metavar = "gbk/fasta", 
                        help = "The format of genome file (gbk or fasta)")
    parser.add_argument("-o", "--outdir", type = str, required = False, default = os.getcwd(), 
                        help = 'The output directory to store the results (default: current directory)')
    parser.add_argument("-s", "--score", type = float, required = False, default = 0.5,
                        help = "The mininum prediction score for a TA system.")
    parser.add_argument("-k", "--keep", required = False, action = "store_true", default = False, 
                        help = 'Whether to keep the temporary directory (default: False)')
    parser.add_argument("-e", "--extract_seq", required = False, action = "store_true", default = False, 
                        help = 'Whether to extract predicted TA sequences (default: False)')
    parser.add_argument("--maximum_distance", type = int, required = False, default = 100, 
                        help = 'Maximum intergenic distances (bp) between potential toxin and antitoxin (default: 100)')
    parser.add_argument("--minimum_distance", type = int, required = False, default = -50, 
                        help = 'Minimum intergenic distances (bp) between potential toxin and antitoxin (default: -50)')
    parser.add_argument("--maximum_length_antitoxin", type = int, required = False, default = 300, 
                        help = 'Maximum sequence length (a.a.) of potential antitoxin (default: 300)')
    parser.add_argument("--maximum_length_toxin", type = int, required = False, default = 500, 
                        help = 'Maximum sequence length (a.a.) of potential toxin (default: 500)')

    return parser.parse_args()    


def prepare_input_file(input, format, tmp_dir, maximum_length_antitoxin, maximum_length_toxin, maximum_distance, minimum_distance):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    faa_file = os.path.join(tmp_dir, "sequence.faa")
    fna_file = os.path.join(tmp_dir, "sequence.fna")
    faa_tmp_file = os.path.join(tmp_dir, "sequence_tmp.faa")
    ptt_file = os.path.join(tmp_dir, "sequence.ptt")
    log_file = os.path.join(tmp_dir, "sequence.log")

    # Check format
    if format == "fasta":
        if not is_fasta(input):
            sys.exit("Your input file is not in standard FASTA format.")
    elif format == "gbk":
        if not is_gbk(input):
            sys.exit("Your input file is not in standard GenBank format.")
    else:
        sys.exit("The '-f/--format' parameter must be either gbk or fasta.")

    if format == "gbk":
        # Extract protein sequences and ptt files
        cmd_gbk2faa = "python " + os.path.join(script_dir, "scripts/utils/gbk2faa_locus_tag.py") + " -i " + input + " -o " + faa_file
        cmd_gbk2fna = "python " + os.path.join(script_dir, "scripts/utils/gbk2fna.py") + " " + input + " " + fna_file
        cmd_gbk2ptt = "python " + os.path.join(script_dir, "scripts/utils/gbk2ptt.py") + " -i " + input + " -o " + ptt_file
        os.system(cmd_gbk2faa)
        os.system(cmd_gbk2fna)
        os.system(cmd_gbk2ptt)
        
    elif format == "fasta":
        os.system(f"cp {input} {fna_file}")
        record = next(SeqIO.parse(input, "fasta"), None)
        seq_len = len(record.seq)

        if seq_len < 20000:
            cmd_prodigal = f"{script_dir}/scripts/utils/prodigal_v2.6.3 < {input} > {tmp_dir}/prodigal.out -p meta"
        else:
            cmd_prodigal = f"{script_dir}/scripts/utils/prodigal_v2.6.3 < {input} > {tmp_dir}/prodigal.out"
        subprocess.run(cmd_prodigal, shell=True)
        
        parse_prodigal_into_ptt(f"{tmp_dir}/prodigal.out", ptt_file, input)

        tsv_file = os.path.join(tmp_dir, "sequence.ptt.coords")
        ffa_file = os.path.join(tmp_dir, "sequence.ffa")
        if not os.path.exists(ptt_file):
            finish_file = os.path.join(tmp_dir, "finish")
            with open(finish_file, 'w') as f:
                f.write('No protein detected')
            exit()
        else:
            with open(ptt_file, "r") as pttfile:
                for line in pttfile:
                    if line.startswith("(\d+) proteins"):
                        if int(line.group(1)) == 0:
                            finish_file = os.path.join(tmp_dir, "finish")
                            with open(finish_file, "w") as done:
                                done.write("No protein detected")
                            exit()

        transform_cds_file(ptt_file)
        subprocess.run(f"{script_dir}/scripts/utils/readCDSseq {input} {tsv_file} {ffa_file} >> {log_file}", shell=True)
        subprocess.run(f"transeq -sequence {ffa_file} -outseq {faa_tmp_file}", shell=True)
        rmstar4faa(faa_tmp_file, faa_file)

    # Construct input file for prediction
    seq = SeqIO.parse(faa_file, "fasta")
    dict_seq = {}
    for seq_record in seq:
        dict_seq[seq_record.id] = str(seq_record.seq)

    df_query = pd.DataFrame(columns=['id_1', 'id_2', 'label', 'seq_1', 'seq_2', 'strand', 'coordinates_1', 'coordinates_2', 'gene_1', 'gene_2', 'product_1', 'product_2'])

    # Predict in two orientations
    ptt = pd.read_csv(ptt_file, header = 2, sep = '\t')
    ptt['start'] = ptt['Location'].str.split("\.\.").str[0].astype('int')
    ptt['end'] = ptt['Location'].str.split("\.\.").str[1].astype('int')
    ptt_plus = ptt[ptt['Strand'] == '+']
    ptt_minus = ptt[ptt['Strand'] == '-']
    #ptt_minus = ptt_minus.iloc[::-1]
    label = 1
    for df in [ptt_plus, ptt_minus]:
        df.index = range(0, df.shape[0])
        for i in range(0, df.shape[0] - 1):
            #for j in range(i, i+1):
            j = i + 1
            length_1 = df.loc[i, 'Length']
            length_2 = df.loc[j, 'Length']
            pid_1 = df.loc[i, 'PID']
            pid_2 = df.loc[j, 'PID']
            distance = df.loc[j, 'start'] - df.loc[i, 'end'] + 1
            if length_1 >= 30 and length_1 <= maximum_length_antitoxin and length_2 >= 30 and length_2 <= maximum_length_toxin and distance <= maximum_distance and distance >= minimum_distance:
                locus_1 = df.loc[i, 'Synonym']
                locus_2 = df.loc[j, 'Synonym']
                strand = df.loc[i, 'Strand']
                gene_1 = df.loc[i, 'Gene']
                gene_2 = df.loc[j, 'Gene']
                coordinates_1 = df.loc[i, 'Location']
                coordinates_2 = df.loc[j, 'Location']
                product_1 = df.loc[i, 'Product']
                product_2 = df.loc[j, 'Product']
                df_query.loc[df_query.shape[0]] = [locus_1, locus_2, label, dict_seq[locus_1], dict_seq[locus_2], strand, coordinates_1, coordinates_2, gene_1, gene_2, product_1, product_2]
            
    # Add reverse
    df_query_seq_reverse = df_query.copy(deep = True)
    df_query_seq_reverse = df_query_seq_reverse[['id_2', 'id_1', 'label', 'seq_2', 'seq_1', 'strand', 'coordinates_2', 'coordinates_1', 'gene_2', 'gene_1', 'product_2', 'product_1']]
    df_query_seq_reverse.columns = df_query.columns
    df_query = pd.concat([df_query, df_query_seq_reverse], axis=0)
    df_query.index = range(0, df_query.shape[0])        
    
    df_query.to_csv(os.path.join(tmp_dir, "test.tsv"), sep = '\t', index = False, header = False)
    return df_query


def parse_prodigal_into_ptt(infile, outfile, fasta_file):    
    seqio_obj = SeqIO.parse(fasta_file, 'fasta')
    seq_obj = next(seqio_obj)
    display_name = seq_obj.name
    desc = seq_obj.description
    seq_length = len(seq_obj.seq)
    title = desc if not desc == "" else display_name
    
    with open(infile) as infile:
        i = 0
        content = ""
        for line in infile:
            res = re.search(r"^\s+CDS\s+(\d+)\.\.(\d+)", line)
            res_complement = re.search(r"^\s+CDS\s+complement\((\d+)\.\.(\d+)\)", line)
            if res_complement:
                start = int(res_complement.group(1))
                end = int(res_complement.group(2))
                strand = "-"
            elif res:
                start = int(res.group(1))
                end = int(res.group(2))
                strand = "+"
            else:
                continue
            i += 1
            length = (end - start + 1) // 3 - 1
            gi = "ORF" + str(i)
            orf = "ORF" + str(i)
            content += f"{start}..{end}\t{strand}\t{length}\t{gi}\t-\t{orf}\t-\t-\tPredicted ORF by PRODIGAL Version 1.20\n"

    with open(outfile, "w") as outfile:
        outfile.write(f"{title} - 1..{seq_length}\n")
        outfile.write(f"{i} proteins\n")
        outfile.write("Location\tStrand\tLength\tPID\tGene\tSynonym\tCode\tCOG\tProduct\n")
        outfile.write(content)


def rmstar4faa(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    with open(output_file, 'w') as f:
        for line in lines:
            line = re.sub('\*', '', line)
            line = re.sub('_1', '', line)
            f.write(line)


def transform_cds_file(infile):
    tempfile = infile + ".coords"
    start = []
    stop = []
    strand = []
    name = []
    i = 0
    CDSNum = 0

    with open(infile, 'r') as input_file:
        for line_no, line in enumerate(input_file, start=1):
            if line_no > 2 and re.match(r'[0-9]+\.\.[0-9]+', line):
                fields = line.split("\t")
                m = re.search(r'([0-9]+)..([0-9]+)', fields[0])
                if m:
                    start.append(m.group(1))
                    stop.append(m.group(2))
                    strand.append(fields[1])
                    name.append(fields[3])
                    if len(fields) > 6:
                        name[CDSNum] += "\n"
                    CDSNum += 1

    with open(tempfile, 'w') as outfile:
        for i in range(CDSNum):
            outfile.write("{}\t{}\t{}\t{}".format(start[i], stop[i], strand[i], name[i]))


def is_fasta(file):
    with open(file, "r") as handle:
        fasta = SeqIO.parse(handle, "fasta")
        return any(fasta)  # False when `fasta` is empty, i.e. wasn't a FASTA file


def is_gbk(file):
    with open(file, "r") as handle:
        gbk = SeqIO.parse(handle, "genbank")
        return any(gbk)  # False when `gbk` is empty, i.e. wasn't a GenBank file


def parse_result(result_file, df_query, score):
    df_result = pd.read_csv(result_file, header = 0, sep = '\t')
    df_result = df_result[(df_result['prediction'] == 1) & (df_result['probability'] >= score)]
    df_result = df_result.merge(df_query, how = 'left', left_index = True, right_index = True)
    #df_result.drop(columns = ['seq_1', 'seq_2'], inplace = True)
    df_result.index = range(0, df_result.shape[0])
    return df_result


def extract_protein_sequence(df_result, score, outdir):
    df_result = df_result[(df_result['prediction'] == 1) & (df_result['probability'] >= score)]
    seq = []

    for i in range(0, df_result.shape[0]):
        ta_id = df_result.loc[i, 'index']
        id_1 = df_result.loc[i, 'id_1']
        id_2 = df_result.loc[i, 'id_2']

        record_1 = SeqRecord(
            Seq(df_result.loc[i, 'seq_1']),
            id=f"TA_{ta_id}_antitoxin-{id_1}",
            description=""
        )
        record_2 = SeqRecord(
            Seq(df_result.loc[i, 'seq_2']),
            id=f"TA_{ta_id}_toxin-{id_2}",
            description=""
        )
        seq.append(record_1)
        seq.append(record_2)
        SeqIO.write(record_1, os.path.join(outdir, f"protein_seq_TA_{ta_id}_AT.fasta"), "fasta")
        SeqIO.write(record_2, os.path.join(outdir, f"protein_seq_TA_{ta_id}_T.fasta"), "fasta")

    return seq


def extract_nucleotide_sequence(df_result, score, tmp_dir, outdir):
    df_result = df_result[(df_result['prediction'] == 1) & (df_result['probability'] >= score)]
    genome_file = os.path.join(tmp_dir, 'sequence.fna')
    
    genome_record = SeqIO.read(genome_file, "fasta")
    genome_seq = str(genome_record.seq)
    
    seq_records = []
    
    for i in range(0, df_result.shape[0]):
        row = df_result.iloc[i]
        strand = row['strand']
        ta_id = row['index']

        coords_1 = row['coordinates_1']
        id_1 = row['id_1']        
        
        gene_seq_1 = extract_gene_sequence(genome_seq, coords_1, strand)
        record_1 = SeqRecord(
                Seq(gene_seq_1),
                id=f"TA_{ta_id}_antitoxin-{id_1}",
                description=""
            )
        seq_records.append(record_1)
        
        coords_2 = row['coordinates_2']
        id_2 = row['id_2']
        
        gene_seq_2 = extract_gene_sequence(genome_seq, coords_2, strand)
        record_2 = SeqRecord(
                Seq(gene_seq_2),
                id=f"TA_{ta_id}_toxin-{id_2}",
                description=""
            )
        seq_records.append(record_2)

        SeqIO.write(record_1, os.path.join(outdir, f"nucl_seq_TA_{ta_id}_AT.fasta"), "fasta")
        SeqIO.write(record_2, os.path.join(outdir, f"nucl_seq_TA_{ta_id}_T.fasta"), "fasta")

    return seq_records


def extract_gene_sequence(genome_seq, coordinates, strand):
    start, end = map(int, coordinates.split('..'))
    gene_seq = genome_seq[start-1:end]
    if strand == '-':
        gene_seq = str(Seq(gene_seq).reverse_complement())

    return gene_seq


def main():
    args = parse_args()

    tmp_dir = os.path.join(args.outdir, "tmp")
    if not os.path.exists(args.outdir):
        try:
            os.mkdir(args.outdir)
        except Exception as ex:
            print(ex)
    if not os.path.exists(tmp_dir):
        try:
            os.mkdir(tmp_dir)
        except Exception as ex:
            print(ex)

    # Extract protein sequences and prepare for the files for prediction
    print("### Prepare for input ###")
    df_query = prepare_input_file(args.input, args.format, tmp_dir, args.maximum_length_antitoxin, args.maximum_length_toxin, args.maximum_distance, args.minimum_distance)
    if (len(df_query) == 0):
        sys.exit("No candidates are available.")

    # Run prediction
    print("### Running prediction ###")
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = args.model
    model = 'esm2model'
    pred_script_path = os.path.join(script_dir, "train.py")
    cmd_run_prediction = "python {} \
                        --model {} \
                        --model_dir {} \
                        --batch_size 128 \
                        --num_workers 32 \
                        --seed 42 \
                        --data_dir {} \
                        --log_dir {} \
                        --dropout_rate 0.4 \
                        --num_layers 0 \
                        --save_attn \
                        --mode predict".format(
                            pred_script_path,
                            model,
                            model_dir,
                            tmp_dir,
                            tmp_dir
                        )
    os.system(cmd_run_prediction)

    # Parse results
    result_file = os.path.join(tmp_dir, "test_result.txt")
    df_result = parse_result(result_file, df_query, args.score)
    if df_result.shape[0] == 0:
        sys.exit("No TA system was predicted.")
    # Order by prediction score
    df_result.sort_values(by='probability', ascending=False, inplace=True)
    df_result['index'] = range(1, len(df_result) + 1)
    df_result.to_csv(os.path.join(args.outdir, "result.csv"), index = False)

    # Extract predicted sequences
    ta_protein_seq = extract_protein_sequence(df_result, args.score, args.outdir)
    protein_seq_output = os.path.join(args.outdir, "protein_seq.fasta")
    SeqIO.write(ta_protein_seq, protein_seq_output, "fasta")

    ta_nucleotide_seq = extract_nucleotide_sequence(df_result, args.score, tmp_dir, args.outdir)
    nucleotide_seq_output = os.path.join(args.outdir, "nucleotide_seq.fasta")
    SeqIO.write(ta_nucleotide_seq, nucleotide_seq_output, "fasta")

    # Remove tmp dir
    if not args.keep:
        shutil.rmtree(tmp_dir)

    print("Prediction is done!")
    

if __name__ == "__main__":
    main()
