#-------------------------------------------------------------------------------
#  Name:       Extract dense trajectory features for ChaLearn LAP 2014 
#  Uses:       1. ChaLearn startup code
#              2. Dense trajectory code from LEAR, INRIA
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Name:        Starting Kit for ChaLearn LAP 2014 Track2
# Purpose:     Show basic functionality of provided code
#
# Author:      Xavier Baro
#
# Created:     19/02/2014
# Copyright:   (c) Chalearn LAP 2014
# Licence:     GPL3
#-------------------------------------------------------------------------------
import sys, os, os.path,random,numpy,zipfile
import cv2, subprocess
from subprocess import Popen, PIPE
from shutil import copyfile
import scipy
import scipy.io
import scipy.stats
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from time import *
import pickle

from ChalearnLAPEvaluation import evalAction,exportGT_Action
from ChalearnLAPSample import ActionSample

from DenseTrajDesc import *
from DenseTrajBOW import *
from getNoActionList import *

from collections import defaultdict


denseTrajectoryExe = '../thirdparty/dense_trajectory_release_v1.2/release/DenseTrack'
trainingVideos = 'trainingVideos'
denseFeatures = 'denseFeatures'

NumActions = 11

descHOG = []
bowTraj = DenseTrajBOW()
D = DenseTrajDesc() 
model = SVC()

def main():
    """ Main script. Show how to perform all competition steps """
    # Data folder (Training data)
    data='./data/';
    # Train folder (output)
    outTrain='./training/train/'
    # Test folder (output)
    outTest='./training/test/'
    # Predictions folder (output)
    outPred='./pred/';
    # Ground truth folder (output)
    outGT='./gt/';
    # Submision folder (output)
    outSubmision='./submision/'

    # Divide data into train and test
    createDataSets(data,outTrain,outTest,0.3);

    # Learn your model
#    if os.path.exists("model.npy"):
#        model=numpy.load("model.npy");
#    else:
#        model=learnModel(outTrain);
#        numpy.save("model",model);

#    if os.path.exists("model.pkl"):
#        model=pickle.load("model.pkl");
#    else:
#        model=learnModel(outTrain);
#        pickle.dump(model,'model.pkl');

    model=learnModel(outTrain);

    # Predict over test dataset
    predict(model,outTest,outPred);

    # Create evaluation gt from labeled data
    exportGT_Action(outTest,outGT);

    # Evaluate your predictions
    score=evalAction(outPred, outGT);
    print("The score for this prediction is " + "{:.12f}".format(score));

    # Prepare submision file (only for validation and final evaluation data sets)
    createSubmisionFile(outPred,outSubmision);

def createDataSets(dataPath,trainPath,testPath,testPer):
    """ Divide input samples into Train and Test sets """
    # Get the data files
    fileList = os.listdir(dataPath);

    # Filter input files (only ZIP files)
    sampleList=[];
    for file in fileList:
        if file.endswith(".zip"):
            sampleList.append(file);

    # Calculate the number of samples for each data set
    numSamples=len(sampleList);
    numTest=round(numSamples*testPer);
    numTrain=numSamples-numTest;

    # Create a random permutation of the samples
    #random.shuffle(sampleList);

    # Create the output partitions
    if os.path.exists(trainPath):
        trainFileList = os.listdir(trainPath);
        for file in trainFileList:
            os.remove(os.path.join(trainPath,file));
    else:
        os.makedirs(trainPath);

    # Create the output partitions
    if os.path.exists(testPath):
        testFileList = os.listdir(testPath);
        for file in testFileList:
            os.remove(os.path.join(testPath,file));
    else:
        os.makedirs(testPath);

    # Copy the files
    count=0;
    for file in sampleList:
        if count<numTrain:
            copyfile(os.path.join(dataPath,file), os.path.join(trainPath,file));
        else:
            copyfile(os.path.join(dataPath,file), os.path.join(testPath,file));
        count=count+1;

def learnModel(data):
    """ Access the sample information to learn a model. """
    print("Learning the model");
    # Get the list of training samples
    samples=os.listdir(data);
    #samples = ['Seq01.zip', 'Seq03.zip', 'Seq04.zip'];  # Hard coded for experiments

    # Initialize the model
    #model=[];
    model = None

    yy = []
    ff = []
    wordIDs = None
    words = None
    t1 = time()

    dataHOG  = None
    fMap = {}
    fMapS = []
    featureVectorNum = 0
    fMap = defaultdict(list)

    print 'Training Set: ', samples

    if not os.path.exists("Features.mat"):
        # Access to each sample
        for file in samples:
            if not file.endswith(".zip"):
                continue;
            print("\t Processing file " + file)

            # Create the object to access the sample
            smp=ActionSample(os.path.join(data,file));

            # ###############################################
            # USE Ground Truth information to learn the model
            # ###############################################

            # Get the list of actions for this frame
            actionList=smp.getActions();

            # Iterate for each action in this sample
            proc = {}
            stdout = {}
            stderr = {}


            print actionList
            sortedActionList  = sorted(actionList)

            noActionList = getNoActionList(sortedActionList)
            for nal in noActionList:
                actionList.append([NumActions+1,nal[0],nal[1]])
            # continue

            for action in actionList:
                # Get the action ID, and start and end frames for the action
                actionID,startFrame,endFrame=action;
                print 'Action: ', actionID, '\t', 'startFrame: ', startFrame, '\t', 'endFrame:', endFrame
                #output = subprocess.check_output('/bin/ps')
                #print output
                #print denseTrajectoryExe, os.path.splitext(file)[0], startFrame, endFrame
                #cmd = []
                #cmd.append(denseTrajectoryExe)
                #seqn = os.path.splitext(file)[0];
                #cmd.append('training/train/%s/%s_color.mp4' % (seqn,seqn))
                #cmd.append('-S')
                #cmd.append(str(startFrame))
                #cmd.append('-E')
                #cmd.append(str(endFrame))
                #print cmd
                #proc[actionID] = Popen(cmd, stdout=PIPE, stderr=PIPE)
                #stdout[actionID], stderr[actionID] = proc[actionID].communicate()
                #for line in stdout[actionID]:
                #    print line,
                # NOTE: We use random predictions on this example, therefore, nothing is done with the image. No model is learnt.
                # Iterate frame by frame to get the information to learn the model
                seqn = os.path.splitext(file)[0]
                actionFileName = "output_%s_%d_frame%d-%d.avi" % (seqn,actionID,startFrame,endFrame)
                actionFileFullName = "%s%s%s%s%s" % (trainingVideos,os.path.sep,seqn, os.path.sep, actionFileName)
                featureFileName = "densetr_%s_%d_frame%d-%d.txt" % (seqn,actionID,startFrame,endFrame)
                featureFileFullName = "%s%s%s%s%s" % (denseFeatures,os.path.sep,seqn, os.path.sep, featureFileName)

                #if not os.path.exists(featureFileFullName):
                #    continue

                if not os.path.exists(actionFileFullName):
                    print actionFileFullName + ' not present'
                    continue

                if os.path.exists(featureFileFullName) and os.path.getsize(featureFileFullName) > 0:
                    print featureFileFullName, ' exists'
                elif endFrame - startFrame + 1 <= 15:
                    print featureFileFullName, 'too small' 
                else:
                    fout = open(featureFileFullName,"w")
                    cmd = []
                    cmd.append(denseTrajectoryExe)
                    seqn = os.path.splitext(file)[0];
                    cmd.append(actionFileFullName)
                    print cmd
                    proc[actionID] = Popen(cmd, stdout=PIPE, stderr=PIPE)
                    #proc[actionID].stdout.flush()
                    #proc[actionID].stderr.flush()
                    stdout[actionID], stderr[actionID] = proc[actionID].communicate()
                    fout.write(stdout[actionID])
                    fout.close()
                #if not os.path.exists('Features.mat'):
                fin = open(featureFileFullName,"r")
                for line in fin:
                    D.read(line)
                    #print 'featureVectorNum: ', featureVectorNum
                    fMap[(actionID, startFrame, endFrame)].append(featureVectorNum)
                    descHOG.append(D.HOG)
                    fMapSTuple = (actionID, startFrame, endFrame, featureVectorNum)
                    fMapS.append(fMapSTuple)
                    featureVectorNum = featureVectorNum + 1 
                    #yy.append(actionID)
                    #ff.append(D.frameNum)
                    #break
                fin.close()
            #y = numpy.array(yy)
            #frameNum = numpy.array(ff)
               
                #break # TODO: remove
                #for numFrame in range(startFrame,endFrame):
                    # Get the image for this frame
                    #image=smp.getRGB(numFrame);
                    
                    #img = cv2.cv.fromarray(image)
                    #print type(img)
                    #print actionName
                    #cv2.imshow(actionName,image)
                    #cv2.waitKey(10)
            # ###############################################
            # Remove the sample object
            del smp;
            #break # TODO: remove
        if not os.path.exists('Features.mat'):
            if descHOG:
                dataHOG = scipy.vstack(tuple(descHOG))
                fMapSr = scipy.vstack(tuple(fMapS))
                #scipy.io.savemat('Features.mat', {'frameNum':frameNum,'y':y, 'HOG':dataHOG}, format='5')
                scipy.io.savemat('Features.mat', {'fMap':fMapSr, 'HOG':dataHOG}, format='5')

    else:
        dct = {}
        print 'Loading pre calculated features'
        scipy.io.loadmat('Features.mat',dct)
        #y = dct['y']
        dataHOG = dct['HOG']
        fMapSr = dct['fMap']
        for t in fMapSr:
            fMap[(t[0],t[1],t[2])].append(t[3])
        #frameNum = dct['frameNum']

    t2 = time()
    print 'Dense Trajectory Feature Extraction: %f seconds' % (t2-t1) 

    # Extract words
    if not os.path.exists("BOWFeatures.mat"):
        bowTraj.build(dataHOG,None,None,None)
        wordIDs = bowTraj.bowHOG.pred_labels
        words  = bowTraj.bowHOG.centroids
        #print wordIDs # nearest centroid for word 
        #print words   # centroids
        #t3 = time()
        #$print 'BoW build : %f seconds' % (t3-t2)
        #X = bowTraj.calcFeatures(dataHOG,None,None,None)
        #t4 = time()
        scipy.io.savemat('BOWFeatures.mat', {'words':words,'wordIDs':wordIDs}, format='5')
    else:
        dct2 = {}
        dct2 = scipy.io.loadmat('BOWFeatures.mat')
        wordIDs = dct2['wordIDs']
        words = dct2['words']  #centroids

    print 'words.shape', words.shape
    print 'wordIDs.shape', wordIDs.shape

    t3 = time()
    print 'Quantization into words : %f seconds' % (t3-t2)

    # Now we create feature vectors
    print 'Creating feature vectors'
    XX = []
    yy = []
    print 'Training Set: ', samples
    for file in samples:
        if not file.endswith(".zip"):
            continue;
        print("\t Processing file " + file)

        # Create the object to access the sample
        smp=ActionSample(os.path.join(data,file));

        # Get the list of actions for this frame
        actionList=smp.getActions();
        noActionList = getNoActionList(sorted(actionList))
        for nal in noActionList:
            actionList.append([NumActions+1,nal[0],nal[1]])

        cnt = 0
        for action in actionList:
            cnt = cnt + 1 
            # Get the action ID, and start and end frames for the action
            actionID,startFrame,endFrame=action;
            print 'PASS 2: ',  cnt, ' : ', 'Action: ', actionID, '\t', 'startFrame: ', startFrame, '\t', 'endFrame:', endFrame
            h = numpy.zeros(bowTraj.vocszHOG)
            seqn = os.path.splitext(file)[0]
            featureFileName = "densetr_%s_%d_frame%d-%d.txt" % (seqn,actionID,startFrame,endFrame)
            featureFileFullName = "%s%s%s%s%s" % (denseFeatures,os.path.sep,seqn, os.path.sep, featureFileName)
            if not os.path.exists(featureFileFullName):
                print featureFileFullName, ' does not exist' 
                yy.append(actionID)
                XX.append(h)
                continue
            htot = 0
            if (actionID,startFrame,endFrame) in fMap:
               # print (actionID,startFrame,endFrame), fMap[(actionID,startFrame,endFrame)]
                for fID in  fMap[(actionID,startFrame,endFrame)]:
                    idx = wordIDs[fID]
                    h[idx] = h[idx] + 1
                    htot = htot + 1
            if htot > 0:
                h = (1.0 / float(htot)) * h
                #print h
            yy.append(actionID)
            XX.append(h)
    X = scipy.vstack(tuple(XX))
    y = numpy.array(yy)
    #print X
    #print y
    #X = bowTraj.calcFeatures(dataActionHOG,None,None,None)
            
    t4 = time()
    print 'BoW histogram creation for training samples', (t4-t3)
    
   # sys.exit(0)

    #  Create chi squared SVM kernel model

    clf = SVC(kernel=chi2_kernel)
    clf.fit(X,y)
    print clf
    t5 = time()
    print 'SVM train : %f seconds', (t5-t4)

        #numpy.savez('model', X=X, y=y, clf=clf)
        #scipy.io.savemat('model.mat', {'X':X,'y':y,'clf':clf}, format='5')
    model = clf;
#    # Return the model
    return model;

def predict(model,data,output):
    """ Access the sample information to predict the pose. """

    actionID = 0 #initialize
    # Get the list of training samples
    samples=os.listdir(data);
    print samples

    # Access to each sample
    for file in samples:
        # Create the object to access the sample
        smp=ActionSample(os.path.join(data,file));
        print file

        # Create a random set of actions for this sample
        numFrame=0;
        pred=[];
        seqn = os.path.splitext(file)[0];
        while numFrame<smp.getNumFrames():
            # Generate an initial displacement
            #start=numFrame+random.randint(1,100);
            start = numFrame

            # Generate the action duration
            #end=min(start+random.randint(10,100),smp.getNumFrames());
            end = min(numFrame+30,smp.getNumFrames())


            actionFileName = "test_%s_frame%d-%d.avi" % (seqn,start,end)
            actionFileFullName = "%s%s%s%s%s" % (trainingVideos,os.path.sep,seqn, os.path.sep, actionFileName)

            w=int(smp.rgb.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
            h=int(smp.rgb.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
            fps=int(smp.rgb.get(cv2.cv.CV_CAP_PROP_FPS))
        
            if os.path.exists(actionFileFullName) and os.path.getsize(actionFileFullName) > 0:
                print actionFileFullName, ' exists'
            else:
                out = cv2.VideoWriter(actionFileFullName,cv2.cv.CV_FOURCC('X','V','I','D'),fps,(w,h))
                for n in range(start,end):
                    image=smp.getRGB(n+1);
                    #print type(image)
                    #print n
                    #img=cv2.cv.fromarray(image)
                    #cv2.imshow('my',image)
                    #cv2.waitKey(10)
                    #print type(img)
                    out.write(image)
                out.release()

            featureFileName = "densetr_%s_frame%d-%d.txt" % (seqn,start,end)
            featureFileFullName = "%s%s%s%s%s" % (denseFeatures,os.path.sep,seqn, os.path.sep, featureFileName)

            #if not os.path.exists(featureFileFullName):
            #    continue

            if os.path.exists(featureFileFullName) and os.path.getsize(featureFileFullName) > 0:
                print featureFileFullName, ' exists'
            else:
                fout = open(featureFileFullName,"w")
                print featureFileFullName
                cmd = []
                cmd.append(denseTrajectoryExe)
                seqn = os.path.splitext(file)[0];
                cmd.append(actionFileFullName)
                print cmd
                proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
                #proc[actionID].stdout.flush()
                #proc[actionID].stderr.flush()
                stdout, stderr = proc.communicate()
                fout.write(stdout)
                fout.close()

            hst = numpy.zeros(bowTraj.vocszHOG)
            if not os.path.exists(featureFileFullName):
                print featureFileFullName, ' not found'
                continue
            htot = 0
            fin = open(featureFileFullName,"r")
            for line in fin:
                D.read(line)
                #descHOG.append(D.HOG)
                #X = bowTraj.calcFeatures(scipy.vstack(tuple(D.HOG)))
                #actionID = model.predict(X)
                idx = bowTraj.bowHOG.kmeans.predict(D.HOG)
                hst[idx] = hst[idx] + 1
                htot = htot + 1
            if htot > 0:
                hst = (1.0 / float(htot)) * hst


            fin.close()
            print 'Retrieved model:'
            print model
            actionID = model.predict(hst)[0]
            print hst
            print 'Predicted: ', actionID
          

            # Generate the action ID
            #actionID=random.randint(1,11);


            # Check if the number of frames are correct
            if start<end-1 and end<smp.getNumFrames():
                # Store the prediction
                pred.append([actionID,start,end])

            # Move ahead
            #numFrame=end+1;
            numFrame=start+15;



        # Store the prediction
        smp.exportPredictions(pred,output);

        # Remove the sample object
        del smp;

def createSubmisionFile(predictionsPath,submisionPath):
    """ Create the submission file, ready to be submited to Codalab. """

    # Create the output path and remove any old file
    if os.path.exists(submisionPath):
        oldFileList = os.listdir(submisionPath);
        for file in oldFileList:
            os.remove(os.path.join(submisionPath,file));
    else:
        os.makedirs(submisionPath);

    # Create a ZIP with all files in the predictions path
    zipf = zipfile.ZipFile(os.path.join(submisionPath,'Submission.zip'), 'w');
    for root, dirs, files in os.walk(predictionsPath):
        for file in files:
            zipf.write(os.path.join(root, file), file, zipfile.ZIP_DEFLATED);
    zipf.close()


if __name__ == '__main__':
    main()
