import baofast
import math
import numpy as np
from astropy.io import fits
import sys

DEGTORAD = math.pi / 180

class combinatorial(baofast.routine):

    def __call__(self):
        binsDec = DEGTORAD * self.getPre("centerDec").data["binCenter"]
        binsRA = DEGTORAD * self.getPre("centerRA").data["binCenter"]
        self.rAng = self.getPre("RANG").data

        # calculate and histogram first row dTheta12, weighted with R1*R2

        self.sinDec = np.sin(binsDec)
        self.cosDec = np.cos(binsDec)
        self.prodSinDec = np.multiply.outer(self.sinDec,self.sinDec)
        self.prodCosDec = np.multiply.outer(self.cosDec,self.cosDec)
        self.cosDeltaRA = np.cos(np.subtract.outer(binsRA,binsRA))

        splits = range(0, len(self.rAng), self.config.chunkSize())
        slices = [slice(i,j) for i,j in zip(splits,splits[1:]+[None])]

        binsTheta = 1000
        cosThetaEdges = np.linspace(-1,1,1+binsTheta)
        
        typeRR = np.int64
        RR = np.zeros(len(cosThetaEdges)-1, dtype=typeRR)

        chunks = [(slices[i],jSlice) for i in range(len(slices)) for jSlice in slices[i:]]
        trunc = 53
        print "There are %d chunks" % len(chunks)
        if trunc: print "Truncating after %d chunks" % trunc

        for slice1,slice2 in chunks[:trunc]:
            chunkCT = self.cosThetaChunk(slice1, slice2)
            countcount = np.multiply.outer(self.rAng["count"][slice1], self.rAng["count"][slice2]).astype(typeRR)
            if slice1 != slice2: countcount *= 2 # fill histogram with twice-weights
            frq,outEdges = np.histogram( chunkCT, binsTheta, (-1,1), weights = countcount) # uniform binning faster to histogram than iterable binning
            RR += frq
            print '.',
            sys.stdout.flush()
        print
        RR /= 2
        outFile = open("points.txt","w")
        for i,j in zip(cosThetaEdges, RR): print>>outFile, i,j
                
    def cosThetaChunk(self, slice1, slice2):
        return (self.cosDeltaRA[self.rAng["binRA"][slice1]][ :,self.rAng["binRA"][slice2]] *
                self.prodCosDec[self.rAng["binDec"][slice1]][:,self.rAng["binDec"][slice2]] +
                self.prodSinDec[self.rAng["binDec"][slice1]][:,self.rAng["binDec"][slice2]] )

    @property
    def inputFileName(self):
        return self.config.stageFileName('preprocessing')

    def getPre(self, name):
        hdulist = fits.open(self.inputFileName)
        return hdulist[name]
        
