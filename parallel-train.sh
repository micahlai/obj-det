#!/bin/bash
~/.bashrc
./beoqueue.pl -d -q -n "iGpu2:1 iGpu8:1 iGpu13:1 iGpu15:1 iGpu21:1 iGpu25:1" -f '/lab/micah/obj-det/filelist.txt' 'bash /lab/micah/obj-det/parallel-script.sh'
# "test1" "test2" "test3" "test4" "test5" "test6"

