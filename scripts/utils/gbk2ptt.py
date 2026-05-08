from argparse import ArgumentParser
from Bio import SeqIO
import pandas as pd
import re

def parse_args():
    parser = ArgumentParser(description = "Convert GenBank file to PTT file")
    parser.add_argument("-i", "--input", type = str, required = True, help = "Input GenBank file (required)")
    parser.add_argument("-o", "--output", type = str, required = True, help = "Output PTT file (required)")
    return parser.parse_args()


def main():
    args = parse_args()
    gb_record = SeqIO.read(args.input, "genbank")
    description = gb_record.description
    source_region = str(gb_record.features[0].location)
    source_region = re.search(r"\[(.+)\]", source_region).group(1)
    source_region = source_region.replace(':', '..')
    description = description + ". - " + source_region

    df = pd.DataFrame(columns=['Location', 'Strand', 'Length', 'PID', 'Gene', 'Synonym', 'Code', 'COG', 'Product'])

    num_protein = 0
    for feature in gb_record.features:
        if feature.type == 'CDS':
            translation = feature.qualifiers.get('translation')
            if not translation or not translation[0]:
                continue
            locus_tag = feature.qualifiers['locus_tag'][0] if "locus_tag" in feature.qualifiers.keys() else "Locus_" + str(num_protein)
            product = feature.qualifiers['product'][0] if "product" in feature.qualifiers.keys() else "-"
            gene = feature.qualifiers['gene'][0] if "gene" in feature.qualifiers.keys() else "-"
            protein_id = feature.qualifiers['protein_id'][0] if "protein_id" in feature.qualifiers.keys() else "Protein_" + str(num_protein)
            location = str(feature.location)
            if re.search("join", location):
                coord_search = re.search(r".+?([0-9]+).+?([0-9]+).+?([0-9]+).+?([0-9]+).+", location)
                coord_list = [int(coord_search.group(1)), int(coord_search.group(2)), int(coord_search.group(3)), int(coord_search.group(4))]
                start, end = min(coord_list) + 1, max(coord_list)
                coordinates = str(start) + ".." + str(end)
                strand = "+" if re.search("\+", location) else "-"
            else:
                strand = re.search(r"\((.+)\)", location).group(1)
                coordinates = re.search(r"\[(.+)\]", location).group(1)
                coordinates = coordinates.replace(':', '..')
                coordinates = coordinates.replace('>', '')
                coordinates = coordinates.replace('<', '')
                coord_search = re.search(r"([0-9]+)\.\.([0-9]+)", coordinates)
                start, end = int(coord_search.group(1)) + 1, int(coord_search.group(2))
                coordinates = str(start) + ".." + str(end)

            length = int((end - start + 1 - 3) / 3)

            df.loc[df.shape[0]] = [coordinates, strand, length, protein_id, gene, locus_tag, "-", "-", product]
            num_protein += 1

    with open(args.output, "w") as file:
        file.write(description + "\n")
        file.write(str(num_protein) + " proteins\n")

    df.to_csv(args.output, index = False, sep = "\t", mode = 'a')
    

if __name__ == "__main__":
    main()