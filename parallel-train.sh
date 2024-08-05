#!/bin/bash
~/.bashrc
./beoqueue.pl -d -q -n "iGpu2:1 iGpu8:1 iGpu13:1 iGpu15:1 iGpu21:1" -f '/lab/micah/obj-det/filelist2.txt' 'bash /lab/micah/obj-det/parallel-script2.sh'

#train with no branching
wait
#./beoqueue.pl -d -q -n "iGpu2:1 iGpu8:1 iGpu13:1 iGpu15:1 iGpu21:1" -f '/lab/micah/obj-det/filelist2.txt' 'bash /lab/micah/obj-det/parallel-script2.sh'


