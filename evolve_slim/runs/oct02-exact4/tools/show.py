import sys,re
for f in sys.argv[1:]:
    l=open(f).read().strip().split('\n')[-1]
    st=[int(x) for x in re.findall(r'np.int64\((-?\d+)\)',l)]
    if len(st)<16: print(f, l[:300]); continue
    nlr=st[1]-st[4]-st[15]-st[13]
    print(f.split('/')[-1], l[:75])
    print('   leaves',st[0],'P1surv',st[1],'KOSdisc',st[13],'K2disc',st[4],'Krdisc',st[15],'LR',nlr,'fam',st[5],'famdisc %.3g'%st[6],'KOS s %.1f'%(st[12]/1e9),'K2/Kr s %.1f'%(st[8]/1e9),'leafLR s %.1f'%(st[9]/1e9),'fam s %.1f'%(st[10]/1e9), 'P1pass s %.1f'%(st[11]/1e9), 'recount s %.1f'%(st[14]/1e9))
    if len(st) >= 19:
        print('   node-K s %.1f push+pack s %.1f P12 s %.1f' % (st[16]/1e9, st[17]/1e9, st[18]/1e9))
    if len(st) >= 21:
        print('   leafLR(robust_lr) s %.1f  B&B s %.1f  B&B supports %d nodes %d' % (st[19]/1e9, st[20]/1e9, st[2], st[3]))
