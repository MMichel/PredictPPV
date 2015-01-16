import sys

from parsing import parse_psipred



def main(infile):
    
    #ss_str = parse_psipred.horizontal(infile)
    ss_str = parse_psipred.vertical(infile)
    seq_len = len(ss_str)

    print ss_str
    ss_counts = [ss_str.count('H'), ss_str.count('E'), ss_str.count('C')]
    ss_content = [count/float(seq_len) for count in ss_counts]

    return ss_content



if __name__ == "__main__":

    #all_freq = []
    if sys.argv[-1].endswith("horiz") or sys.argv[-1].endswith("ss") or sys.argv[-1].endswith("ss2"):
        sys.stderr.write("It seems you forgot to specify an output file path.\nUsage: python %s <psipred_horiz file 1, ..., n> <outfile>")
        sys.exit(1)
    outfile = open(sys.argv[-1], 'w')

    for infile_name in sys.argv[1:-1]:
        print infile_name
        infile = open(infile_name)
        #all_freq.append(main(infile))
        prot = '.'.join(infile_name.split('/')[-1].split('.')[:2])
        ss_content = main(infile)
        outfile.write('%s %s\n' % (prot, ' '.join(map(str, ss_content))))
