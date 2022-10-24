import textgrid

with open("all.TextGrid") as f:
    for i in f:
        u_id=0
        file_name=i.strip()
        # try:
        tg = textgrid.TextGrid.fromFile(file_name)
        for spk in tg:
            #为每一句话提取句子信息
            for utt in spk:
            
                print(file_name[0:-9]+ "_" + str(u_id).zfill(5) + " " + str(utt.minTime) +" " + str(utt.maxTime) +" " + utt.mark )
                u_id=u_id+1
        # except:
        #     None
        #     print(file_name)

# grid_file ="/home4/shahidah/aug_sep-2022/mml-21-sept-2018-mlycs-session4-closetalk-2.TextGrid"
# textgrid.read_textgrid(grid_file,"sa")