from Bio import SeqIO, SeqFeature
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i',  help = 'Input Genbank format file (include GI)')
parser.add_argument('--output', '-o',  help = 'Output faa format file')
args = parser.parse_args()

faa_out = []
gbk = SeqIO.read(str(args.input), "genbank")
num_protein = 0
for gene in gbk.features:
    if (gene.type == "CDS"):
        translation = gene.qualifiers.get('translation')
        if not translation or not translation[0]:
            continue
        #gi = str(gene.qualifiers.get('locus_tag'))
        gi = gene.qualifiers['locus_tag'][0] if "locus_tag" in gene.qualifiers.keys() else "Locus_" + str(num_protein)
        num_protein += 1
        #if (len(gi.split("'")) > 1):
        #    gi = gi.split("'")[1]
        #else:
        #    continue
	# Head line
        header = ('>%s' % gi)
        faa_out.append(header + '\n')
        # Extract amino acid sequences
        protein_seq = ''.join(re.findall(r'[A-Za-z]', str(gene.qualifiers.get('translation'))))
        protein_seq = protein_seq.replace('J', '').replace('Z', '')
        # Insert "\n" per 70 letters into sequences 
        i = 0
        tmp = []
        while True:
            length_protein_seq = len(protein_seq)                
            if ((i + 70) < length_protein_seq):
                tmp.append(protein_seq[i:(i + 70)])
                i = i + 70
            else:
                tmp.append(protein_seq[i:length_protein_seq])
                break
        protein_seq = "\n".join(tmp) + "\n"
        faa_out.append(protein_seq) 

# Output amino acid sequences and DNA sequences
with open(str(args.output), "w") as f:
    f.writelines(faa_out)   
