#!/bin/bash
~/.bashrc

./beoqueue.pl -d -q -n "iGpu21:1 iGpu15:1 iGpu2:1 iGpu8:1 localhost" -f '/lab/micah/obj-det/second-dataset-wave.txt' 'bash /lab/micah/obj-det/parallel-script2 copy.sh'

wait
./commitAndPushAll.sh 2nd-dataset-all-freeze-sets
wait
./beoqueue.pl -d -q -n "iGpu21:1 iGpu15:1 iGpu2:1 iGpu8:1 localhost" -f '/lab/micah/obj-det/alternate-test.txt' 'bash /lab/micah/obj-det/parallel-script2.sh'
wait
./commitAndPushAll.sh alternate-freeze-set-for-1st-wave
wait
./killallmyprocs.pl