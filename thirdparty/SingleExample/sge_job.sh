#!/bin/sh
#$-cwd
#$-N SingleExample
#$-j y
#$-o /nfs/bigeye/sdaptardar/ChaLearn/ChaLearn/thirdparty/SingleExample/log.$JOB_ID.out
#$-e /nfs/bigeye/sdaptardar/ChaLearn/ChaLearn/thirdparty/SingleExample/log.$JOB_ID.err
#$-M sdaptardar@cs.stonybrook.edu
#$-m ea
#$-l hostname=detection.cs.stonybrook.edu
#$-l virtual_free=32G
#$-pe default 2
#$-R y

export LD_LIBRARY_PATH=/opt/matlab_r2010b/bin/glnxa64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
export DISPLAY=localhost:12.0
echo "Starting job: $SGE_TASK_ID"
echo "Output File: /nfs/bigeye/sdaptardar/ChaLearn/ChaLearn/thirdparty/SingleExample/log.${JOB_ID}.${TASK_ID}.out"
echo "Error File:  /nfs/bigeye/sdaptardar/ChaLearn/ChaLearn/thirdparty/SingleExample/log.${JOB_ID}.${TASK_ID}.err"
#matlab -nodesktop -nosplash -singleCompThread < /nfs/bigeye/sdaptardar/ChaLearn/ChaLearn/thirdparty/SingleExample/extract3DLark.m 
matlab -nodesktop -nosplash < /nfs/bigeye/sdaptardar/ChaLearn/ChaLearn/thirdparty/SingleExample/extract3DLark.m 
echo "Ending job: $SGE_TASK_ID"
