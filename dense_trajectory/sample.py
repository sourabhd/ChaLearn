import sys, os, os.path,random,numpy,zipfile
from shutil import copyfile

from ChalearnLAPEvaluation import evalAction,exportGT_Action
from ChalearnLAPSample import ActionSample
import cv2

data='./data/'; 
samples=os.listdir(data);
# Initialize the model
model=[];
#fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Access to each sample
for file in samples:
	 # Create the object to access the sample
     smp=ActionSample(os.path.join(data,file));
     # Get the list of actions for this frame
     actionList=smp.getActions();
     print file
     print smp.getNumFrames()
     seqn=os.path.splitext(file)[0]
     name='trainingVideos/'+seqn
     os.mkdir(name)
     for action in actionList:
        # Get the action ID, and start and end frames for the action
        actionID,startFrame,endFrame=action;
        print startFrame,endFrame
        
       # fourcc=int(smp.rgb.get(cv2.cv.CV_CAP_PROP_FOURCC))
        w=int(smp.rgb.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        h=int(smp.rgb.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        fps=int(smp.rgb.get(cv2.cv.CV_CAP_PROP_FPS))
        
       
 
        out=cv2.VideoWriter(name+'/output_%s_%d_frame%d-%d.avi'%(seqn,actionID,startFrame,endFrame),cv2.cv.CV_FOURCC('X','V','I','D'),fps,(w,h))
     	
        for numFrame in range(startFrame,endFrame):
        	image=smp.getRGB(numFrame);
        	#print type(image)
        	print numFrame
        	img=cv2.cv.fromarray(image)
        	cv2.imshow('my',image)
        	cv2.waitKey(10)
        	#print type(img)
        	out.write(image)
        	

