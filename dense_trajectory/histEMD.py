import cv2, cv
import numpy as np
import scipy as sp
import numpy.linalg


# Source :  https://gist.github.com/oostendo/5901331

#def histEMD(hist1, hist2, hist1weights = [], hist2weights = []):
#    if not len(hist1weights):
#        hist1weights = np.ones(len(hist1))
#    
#    if not len(hist2weights):
#        hist2weights = np.ones(len(hist1))
#    
#    print hist1
#    print hist2
#    print hist1weights
#    print hist2weights
#    a64 = np.dstack((hist1weights, hist1))
#    #a32 = cv2.CreateMat(a64.rows, a64.cols, cv.CV_32FC1)
#    #cv2.Convert(a64, a32)
# 
#    b64 = np.dstack((hist2weights, hist2))
#    #b32 = cv2.CreateMat(b64.rows, b64.cols, cv.CV_32FC1)
#    #cv2.Convert(b64, b32)
#    
#    return cv2.CalcEMD2(,b64,cv2.CV_DIST_L2)
#    #return cv.CalcEMD2(a32,b32,cv.CV_DIST_L2)
#


def histEMD(hist1, hist2,cost):
    
    print hist1
    print hist2

    # Matrix conversion 
    h1_64 = cv.fromarray(hist1.copy())
    h1_32 = cv.CreateMat(h1_64.rows, h1_64.cols, cv.CV_32FC1)
    cv.Convert(h1_64, h1_32)
    h2_64 = cv.fromarray(hist2.copy())
    h2_32 = cv.CreateMat(h2_64.rows, h2_64.cols, cv.CV_32FC1)
    cv.Convert(h2_64, h2_32)

    cost_64 = cv.fromarray(cost)
    cost_32 = cv.CreateMat(cost_64.rows, cost_64.cols, cv.CV_32FC1)
    cv.Convert(cost_64, cost_32)


    idty_64 = cv.fromarray(np.identity(cost_64.rows))
    idty_32 = cv.CreateMat(idty_64.rows, idty_64.cols, cv.CV_32FC1)
    cv.Convert(idty_64, idty_32)

    #return cv.CalcEMD2(signature1=h1_32,signature2=h2_32,distance_type=cv.CV_DIST_USER,cost_matrix=cost_32,distance_func=None)
    return cv.CalcEMD2(signature1=h1_32,signature2=h2_32,distance_type=cv.CV_DIST_C,cost_matrix=idty_32,distance_func=None)



if __name__ == '__main__':
   # A = np.ndarray(shape=(4,1), dtype='float', buffer=np.array([1.,2.,3.,4.],dtype='float'))
   # B = np.ndarray(shape=(4,1),dtype='float', buffer=np.array([1.,2.,3.,4.],dtype='float'))
   # Awt = np.ndarray(shape=(4,1), dtype='float', buffer=np.array([1.,1.,1.,1.],dtype='float'))
   # Bwt = np.ndarray(shape=(4,1),dtype='float', buffer=np.array([1.,1.,1.,1.],dtype='float'))
    A = np.matrix('10.;20.;30.;40.')
    B = np.matrix('10.;20.;30.;40.')
    #cost = np.ndarray(shape=(4,4), dtype='float', buffer=np.array([0., 1., 2., 3.,  1., 0., 1., 2.,  2., 1., 0., 1.,  3., 2., 1., 0. ],dtype='float'))
    cost = np.matrix('0., 1., 2., 3.;  1., 0., 1., 2.;  2., 1., 0., 1.;  3., 2., 1., 0.')
    print  histEMD(A,B,cost) 
