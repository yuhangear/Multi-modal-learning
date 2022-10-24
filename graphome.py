import sys
infile=sys.argv[1]
outfile=sys.argv[2]
outfile=open(outfile,"w")
with open(infile,"r") as f :
    for line in f:
        line=line.strip().split()
        utt=line[0]
        text=line[1:]
        letter=" "
        for word in text:
            i_count=0
            for t in word:
                if i_count==0:
                    letter=letter+" "+t+"_b"
                elif i_count==(len(word)-1):
                    letter=letter+" "+t+"_d"
                else:
                    letter=letter+" "+t+"_i"
                i_count=i_count+1
            i_count=0
        new_text=utt+" "+letter+"\n"
        outfile.writelines(new_text)
outfile.close()

