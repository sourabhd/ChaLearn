from BOW import *

class DenseTrajBOW:

    def __init__(self):
        self.bowHOG = BOW()
        self.bowHOF = BOW()
        self.bowMBFx = BOW()
        self.bowMBFy = BOW()

        self.dimHOG = 96
        self.dimHOF = 108
        self.dimMBFx = 96
        self.dimMBFy = 96

        self.vocszHOG = 128
        self.vocszHOF = 128
        self.vocszMBFx = 128
        self.vocszMBFy = 128

    def build(self,dataHOG,dataHOF,dataMBFx,dataMBFy):
        self.bowHOG.vq(data=dataHOG,voc_size=self.vocszHOG,gt_labels=None)

    def calcFeatures(self,dataHOG,dataHOF,dataMBFx,dataMBFy):
        self.bowHOG.calc_bow_representation(fv=dataHOG)
        return self.bowHOG.bow


