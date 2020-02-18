from __future__ import print_function
import math
import numpy as np
import lasspia as La
from astropy.io import fits
from scipy.integrate import quad
from scipy.sparse import csr_matrix
from lasspia.timing import timedHDU

class integration(La.routine):

    def __call__(self):
        
        grid2d = self.config.grid2D()
        if self.nJobs != None:
            self.hdus.append(self.integrationParameters())
            self.hdus.append(self.binCenters(self.config.binningS(), "centerS") )
            self.hdus.extend(self.tpcf(grid2d = grid2d,compLS = False))
            if grid2d == True:
                self.hdus.append(self.binCenters(self.config.binningSigma(), "centerSigma") )
                self.hdus.append(self.binCenters(self.config.binningPi(), "centerPi") )
            self.writeToFile()
            
        else:
            try:
                self.hdus.append(self.integrationParameters())
                self.hdus.append(self.binCenters(self.config.binningS(), "centerS") )
                self.hdus.extend(self.tpcf(grid2d = grid2d,compLS = True))
                if grid2d == True:
                    self.hdus.append(self.binCenters(self.config.binningSigma(), "centerSigma") )
                    self.hdus.append(self.binCenters(self.config.binningPi(), "centerPi") )
                self.writeToFile()
            except MemoryError as e:
                print(e.__class__.__name__, e, file=self.out)
                print('\n'.join(['Use less memory by integrating via multiple jobs.',
                                 'For example, use options: --nJobs 8 --nCores 1',
                                 'Then combine job outputs: --nJobs 8']),
                      file=self.out)
        return

    def omegasMKL(self): return self.config.omegasMKL()
    def H0(self): return self.config.H0()
    def Pell2(self,mu): return((1/2)*(3*mu**2-1))
    def Pell4(self,mu): return((1/8)*(35*mu**4-30*mu**2+3))

    @timedHDU
    def integrationParameters(self):
        hdu = fits.TableHDU(name='parameters')
        hdu.header['lightspd'] = self.config.lightspeed()
        hdu.header['omegaM'], hdu.header['omegaK'], hdu.header['omegaL'] = self.omegasMKL()
        hdu.header['H0'] = self.H0()
        return hdu

    @timedHDU
    def binCenters(self, binning, name):
        centers = La.utils.centers(self.config.edgesFromBinning(binning))[La.utils.centers(self.config.edgesFromBinning(binning))>0.]
        return fits.BinTableHDU(np.array(centers,dtype = [("binCenter", np.float64)]), name=name)
    
    def returnBins(self, binning, name):
        centers = La.utils.centers(self.config.edgesFromBinning(binning))[La.utils.centers(self.config.edgesFromBinning(binning))>0.]
        return centers

    def lsEst(self,RR,DR,DD,DDe2,nRR,nDR,nDD):
        corr = np.zeros(len(RR))
        corrErr = np.zeros(len(RR))
        for i in range(len(corr)):
            if RR[i] != 0.:
                corr[i] = ((DD[i]/nDD)+(RR[i]/nRR)-2*(DR[i]/nDR))/(RR[i]/nRR)
                corrErr[i] = (np.sqrt(DDe2[i])/nDD)/(RR[i]/nRR)
        return (corr,corrErr)
    
    def lsBundle(self,columns,lsCorrpkg):
        columns.append(fits.Column(name='LSCorr', array = lsCorrpkg[0], format='D'))
        columns.append(fits.Column(name='LSCorrErr', array = lsCorrpkg[1], format='D'))
        return columns

    def muIntegralcorr(self,tpcf_mu_int,sCenters,muCenters,dmu):
        xi_ell0 = np.array([(tpcf_mu_int[i,:]*dmu).sum() for i in range(len(sCenters))])
        xi_ell2 =[(tpcf_mu_int[i,:]*self.Pell2(muCenters)*dmu).sum()*(2*2+1) for i in range(len(sCenters))]
        xi_ell4 = [(tpcf_mu_int[i,:]*self.Pell4(muCenters)*dmu).sum()*(2*4+1) for i in range(len(sCenters))]
        return(xi_ell0,xi_ell2,xi_ell4)
        
    def muIntegralerr(self,tpcf_unc_grid,sCenters,muCenters,dmu):            
        xi_ell0_unc = [np.sqrt(((tpcf_unc_grid[i,:]*dmu)**2).sum()) for i in range(len(sCenters))]
        xi_ell2_unc =[np.sqrt(((tpcf_unc_grid[i,:]*self.Pell2(muCenters)*dmu)**2).sum())*(2*2+1) for i in range(len(sCenters))]
        xi_ell4_unc = [np.sqrt(((tpcf_unc_grid[i,:]*self.Pell4(muCenters)*dmu)**2).sum())*(2*4+1) for i in range(len(sCenters))]
        return(xi_ell0_unc,xi_ell2_unc,xi_ell4_unc)
    
    @timedHDU
    def tpcf(self, grid2d=False, compLS=True):
        self.pdfz = self.getInput('pdfZ').data['probability']
        self.zMask = np.ones(len(self.pdfz)**2, dtype=np.int).reshape(len(self.pdfz),len(self.pdfz))
        self.zMask[:self.config.nBinsMaskZ(),:self.config.nBinsMaskZ()] = 0

        slcT =( slice(None) if self.iJob is None else
                La.slicing.slices(len(self.getInput('centertheta').data),
                                  N=self.nJobs)[self.iJob] )

        def bundleHDU(name, addresses, binning, axes, dropZeros=False, compLS=True, legendre=False):
            if legendre==False:
                rr, dr, dd, dde2 = self.calc(addresses, binning, slcT)
                mask = np.logical_or.reduce([a!=0 for a in [rr, dr, dd]]) if dropZeros else np.full(rr.shape, True, dtype=bool)
                idx_adjust = len(self.returnBins(self.config.binningS(), "centerS"))
                grid = []
                condition = (rr[mask] != 0.)
                for k,iK in zip(axes, np.where(mask)):
                    if len(axes)>1:
                        grid.append(fits.Column(name="i"+k, array=iK[condition]-idx_adjust, format='I'))
                        columns = grid + [fits.Column(name='RR', array=rr[mask][condition], format='D'),
                                          fits.Column(name='DR', array=dr[mask][condition], format='D'),
                                          fits.Column(name='DD', array=dd[mask][condition], format='D'),
                                          fits.Column(name='DDe2', array=dde2[mask][condition], format='D')]
                    else:
                        grid.append(fits.Column(name="i"+k, array=iK, format='I'))
                        columns = grid + [fits.Column(name='RR', array=rr[mask], format='D'),
                                          fits.Column(name='DR', array=dr[mask], format='D'),
                                          fits.Column(name='DD', array=dd[mask], format='D'),
                                          fits.Column(name='DDe2', array=dde2[mask], format='D')]
                
                nRR,nDR,nDD = self.getInput('fTheta').header['NORM'],self.getInput('gThetaZ').header['NORM'],self.getInput('uThetaZZ').header['NORM']
                if compLS==True:
                    columns = self.lsBundle(columns,self.lsEst(rr[mask],dr[mask],dd[mask],dde2[mask],nRR,nDR,nDD))
                    
                hdu = fits.BinTableHDU.from_columns(columns,name=name)
                hdu.header['NORMRR'] = nRR
                hdu.header['NORMDR'] = nDR
                hdu.header['NORMDD'] = nDD
                hdu.header.add_comment("Two-point correlation function for pairs of galaxies (using LS estimator),"+
                                       " by distance" + ("s" if len(axes)>1 else "") + " " +
                                       " and ".join(axes))
                return hdu
            
            if legendre==True:
                rr, dr, dd, dde2 = self.calc(addresses, binning, slcT)
                mask = np.logical_or.reduce([a!=0 for a in [rr, dr, dd]]) if dropZeros else np.full(rr.shape, True, dtype=bool)
                nRR,nDR,nDD = self.getInput('fTheta').header['NORM'],self.getInput('gThetaZ').header['NORM'],self.getInput('uThetaZZ').header['NORM']
                if compLS==False:
                    grid = [fits.Column(name="i"+k, array=iK, format='I')
                            for k,iK in zip(axes, np.where(mask))]
                    columns = grid + [fits.Column(name='RR', array=rr[mask], format='D'),
                                      fits.Column(name='DR', array=dr[mask], format='D'),
                                      fits.Column(name='DD', array=dd[mask], format='D'),
                                      fits.Column(name='DDe2', array=dde2[mask], format='D')]
                    hdu = fits.BinTableHDU.from_columns(columns,name=name)
                    hdu.header['NORMRR'] = nRR
                    hdu.header['NORMDR'] = nDR
                    hdu.header['NORMDD'] = nDD
                    hdu.header.add_comment("Two-point correlation function for pairs of galaxies (using LS estimator),"+
                                           " by distance" + ("s" if len(axes)>1 else "") + " " +
                                           " and ".join(axes))
                    return hdu
                
                if compLS==True:
                    sCenters = self.returnBins(self.config.binningS(), "centerS")
                    muCenters = self.returnBins(self.config.binningMu(), "centerMu")
                    tpcf_mu_int,tpcf_unc_grid = np.zeros((len(sCenters),len(muCenters))),np.zeros((len(sCenters),len(muCenters)))
                    for i in range(len(sCenters)):
                        for j in range(len(muCenters)):
                            if (rr[i,j] != 0.):
                               tpcf_mu_int[i,j] = ((dd[i,j]/nDD)+(rr[i,j]/nRR)-2*(dr[i,j]/nDR))/(rr[i,j]/nRR)
                               tpcf_unc_grid[i,j] = (np.sqrt(dde2[i,j])/nDD)/(rr[i,j]/nRR)
                    del rr, dr, dd, dde2
                    xi_ell0,xi_ell2,xi_ell4 = self.muIntegralcorr(tpcf_mu_int,sCenters,muCenters,muCenters[1]-muCenters[0])
                    xi_ell0_unc,xi_ell2_unc,xi_ell4_unc = self.muIntegralerr(tpcf_unc_grid,sCenters,muCenters,muCenters[1]-muCenters[0])
                    grid = np.asarray(list(range(len(self.returnBins(self.config.binningS(), "centerS")) )))
                    columns = [fits.Column(name='iS', array=grid, format='I')]
                    columns.append(fits.Column(name='ell0Corr', array=xi_ell0, format='D'))
                    columns.append(fits.Column(name='ell0CorrErr', array=xi_ell0_unc, format='D'))
                    columns.append(fits.Column(name='ell2Corr', array=xi_ell2, format='D'))
                    columns.append(fits.Column(name='ell2CorrErr', array=xi_ell2_unc, format='D'))
                    columns.append(fits.Column(name='ell4Corr', array=xi_ell4, format='D'))
                    columns.append(fits.Column(name='ell4CorrErr', array=xi_ell4_unc, format='D'))
                    hdu = fits.BinTableHDU.from_columns(columns,name='Expanded TPCF')
                    hdu.header.add_comment("Legendre multipoles of the two-point correlation function for pairs of galaxies"+
                                           " (using LS estimator) by distance S")
                    return hdu

        sigmaPis = self.sigmaPiGrid(slcT)
        sMus = self.sMuGrid(slcT)
        s = np.sqrt(np.power(sigmaPis,2).sum(axis=-1))

        b2 = self.config.binningDD([self.config.binningS(),
                                    self.config.binningMu()])
        
        b = self.config.binningDD([self.config.binningS()])
        b3 = self.config.binningDD([self.config.binningSigma(),
                                    self.config.binningPi()])
        
        if compLS==True:
            hdu = bundleHDU("TPCF", s, b, ["S"],compLS=True)
            hdu2 = bundleHDU("TPCFLegendre", sMus, b2, ["S", "Mu"], dropZeros=True, compLS=True, legendre=True)
            if grid2d == True:
                hdu3 = bundleHDU("TPCF2D", sigmaPis, b3, ["Sigma", "Pi"], dropZeros=True, compLS=True, legendre=False)
                return [hdu, hdu2, hdu3]
            else: return [hdu, hdu2]
            
        if compLS==False:
            hdu = bundleHDU("TPCF", s, b, ["S"],compLS=False)
            hdu2 = bundleHDU("TPCFLegendre", sMus, b2, ["S", "Mu"], dropZeros=True, compLS=False, legendre=True)
            if grid2d == True:
                hdu3 = bundleHDU("TPCF2D", sigmaPis, b3, ["Sigma", "Pi"], dropZeros=True, compLS=False, legendre=False)
                return [hdu, hdu2, hdu3]
            else: return [hdu, hdu2]

    def sigmaPiGrid(self, slcT):
        '''A cubic grid of (sigma, pi) values
        for pairs of galaxies with coordinates (iTheta, iZ1, iZ2).'''
        Iz = self.zIntegral()
        rOfZ = Iz * (self.config.lightspeed() / self.H0())
        tOfZ = rOfZ * (1 + self.omegasMKL()[1]/6 * Iz**2)

        thetas = self.getInput('centertheta').data['binCenter'][slcT]
        sinT2 = np.sin(thetas/2)
        cosT2 = np.cos(thetas/2)

        sigmas = sinT2[:,None,None] * (tOfZ[None,:,None] + tOfZ[None,None,:])
        pis = cosT2[:,None,None] * np.abs((rOfZ[None,:,None] - rOfZ[None,None,:]))
        return np.stack([sigmas, pis], axis=-1)
    
    def sMuGrid(self, slcT):
        '''A cubic grid of (s, mu) values
        for pairs of galaxies with coordinates (iTheta, iZ1, iZ2).'''
        Iz = self.zIntegral()
        rOfZ = Iz * (self.config.lightspeed() / self.H0())
        tOfZ = rOfZ * (1 + self.omegasMKL()[1]/6 * Iz**2)

        thetas = self.getInput('centertheta').data['binCenter'][slcT]
        sinT2 = np.sin(thetas/2)
        cosT2 = np.cos(thetas/2)

        sigmas = sinT2[:,None,None] * (tOfZ[None,:,None] + tOfZ[None,None,:])
        pis = cosT2[:,None,None] * np.abs((rOfZ[None,:,None] - rOfZ[None,None,:]))
        ss = np.sqrt(sigmas**2+pis**2)
        mus = pis/ss
        return np.stack([ss, mus], axis=-1)

    def calc(self, *args):
        return (self.calcRR(*args),
                self.calcDR(*args),
                self.calcDD(*args, wName='count'),
                self.calcDD(*args, wName='err2'))

    def calcRR(self, addresses, binning, slcT):
        ft = self.getInput('fTheta').data['count'][slcT]
        counts = ft[:,None,None] * self.pdfz[None,:,None] * self.pdfz[None,None,:] * self.zMask[None,:]
        N = counts.size
        D = addresses.size // N
        rr = np.histogramdd(addresses.reshape(N,D), weights=counts.reshape(N), **binning)[0]
        del counts
        return rr

    def calcDR(self, addresses, binning, slcT):
        gtz = self.getInput('gThetaZ').data
        counts = gtz[slcT,:,None] * self.pdfz[None,None,:] * self.zMask[None,:]
        N = counts.size
        D = addresses.size // N
        dr = np.histogramdd(addresses.reshape(N,D), weights=counts.reshape(N), **binning)[0]
        del counts
        return dr

    def calcDD(self, addresses, binning, slcT, wName='count'):
        nZ = addresses.shape[1]

        utzz = self.getInput('uThetaZZ').data
        overflow = utzz['binZdZ'][-1]+1 == nZ**2
        slc = slice(-1 if overflow else None)

        iThetas = utzz['binTheta'][slc]
        mask = (slice(None) if slcT==slice(None) else
                np.logical_and(slcT.start <= iThetas, iThetas < slcT.stop))

        iTh = iThetas[mask] - (slcT.start or 0)
        iZdZ = utzz['binZdZ'][slc][mask]
        iZ = iZdZ // nZ
        diZ = iZdZ % nZ
        iZ2 = iZ + diZ
        counts = utzz[wName][slc][mask] * self.zMask[iZ,iZ2]

        dd = np.histogramdd(addresses[iTh,iZ,iZ2], weights=counts, **binning)[0]
        if overflow and self.iJob in [0,None]:
            dd[-1] = dd[-1] + utzz[wName][-1]
        return dd

    def zIntegral(self):
        zCenters = self.getInput('centerz').data['binCenter']
        zz = zip(np.hstack([[0.],zCenters]), zCenters)
        dIz = [quad(self.integrand, z1, z2, args=self.omegasMKL())[0]
               for z1,z2 in zz]
        return np.cumsum(dIz)

    @staticmethod
    def integrand(z, omegaM, omegaK, omegaLambda):
        return 1./math.sqrt(omegaM * (1+z)**3 +
                            omegaK * (1+z)**2 +
                            omegaLambda)

    @property
    def inputFileName(self):
        return self.config.stageFileName('combinatorial')

    def getInput(self, name):
        hdulist = fits.open(self.inputFileName)
        return hdulist[name]

    def combineOutput(self, jobFiles = None):
        grid2d = self.config.grid2D()
        if not jobFiles:
            jobFiles = [self.outputFileName + self.jobString(iJob)
                        for iJob in range(self.nJobs)]
        with fits.open(jobFiles[0]) as h0:
            for h in ['parameters','centerS']:
                self.hdus.append(h0[h])
            hdu = h0['TPCF']
            hdu2 = h0['TPCFLegendre']
            if grid2d == False:
                cputime = hdu2.header['cputime']
                walltime = hdu2.header['walltime']
                timeKey = 'TPCFLegendre'
            if grid2d == True:
                shape2D = ((h0['TPCF2D'].data['iSigma']).max()+1, (h0['TPCF2D'].data['iPi']).max()+1)
                hdu3 = h0['TPCF2D']
                cputime = hdu3.header['cputime']
                walltime = hdu3.header['walltime']
                timeKey = 'TPCF2D'
                tpcf2d = AdderTPCF2D(h0['TPCF2D'], shape2D)
            shapeleg = (self.config.binningS()['bins'], self.config.binningMu()['bins'])
            tpcfleg = AdderTPCFLegendreGrid(h0['TPCFLegendre'], shapeleg)
            for jF in jobFiles[1:]:
                with fits.open(jF) as jfh:
                    assert np.all(h0['parameters'].header[item] == jfh['parameters'].header[item]
                                  for item in ['lightspd','H0','omegaM','omegaK','omegaL'])
                    for axis in ['centerS']:
                        assert np.all(h0[axis].data['binCenter'] == jfh[axis].data['binCenter'])
                    cputime += jfh[timeKey].header['cputime']
                    walltime += jfh[timeKey].header['walltime']
                    tpcfleg += AdderTPCFLegendreGrid(jfh['TPCFLegendre'], shapeleg)
                    if grid2d == True:
                        tpcf2d += AdderTPCF2D(jfh['TPCF2D'], shape2D)
                    for col in ['RR','DR','DD','DDe2']:
                        hdu.data[col] += jfh['TPCF'].data[col]
            grid = np.asarray(list(range(len(self.returnBins(self.config.binningS(), "centerS")) )))
            hdu1d_cols = [fits.Column(name='iS', array=grid, format='I')] + [fits.Column(name='RR', array=hdu.data['RR'], format='D'),
                                                                             fits.Column(name='DR', array=hdu.data['DR'], format='D'),
                                                                             fits.Column(name='DD', array=hdu.data['DD'], format='D'),
                                                                             fits.Column(name='DDe2', array=hdu.data['DDe2'], format='D')]
            nRR_1d,nDR_1d,nDD_1d = self.getInput('fTheta').header['NORM'],self.getInput('gThetaZ').header['NORM'],self.getInput('uThetaZZ').header['NORM']
            hdu1d_cols = self.lsBundle(hdu1d_cols,self.lsEst(hdu.data['RR'],hdu.data['DR'],hdu.data['DD'],
                                                             hdu.data['DDe2'],nRR_1d,nDR_1d,nDD_1d))
            hdu1d = fits.BinTableHDU.from_columns(hdu1d_cols,name='TPCF')
            hdu1d.header['NORMRR'] = nRR_1d
            hdu1d.header['NORMDR'] = nDR_1d
            hdu1d.header['NORMDD'] = nDD_1d
            hdu1d.header.add_comment("Two-point correlation function for pairs of galaxies"+
                                     " (using LS estimator) by distance S")
            self.hdus.append(hdu1d)
            
            sCenters = self.returnBins(self.config.binningS(), "centerS")
            muCenters = self.returnBins(self.config.binningMu(), "centerMu")
            rr,dr = tpcfleg.fillHDU(h0['TPCFLegendre']).data['RR'],tpcfleg.fillHDU(h0['TPCFLegendre']).data['DR'],
            dd,dde2 = tpcfleg.fillHDU(h0['TPCFLegendre']).data['DD'],tpcfleg.fillHDU(h0['TPCFLegendre']).data['DDe2']
            nRR,nDR = tpcfleg.fillHDU(h0['TPCFLegendre']).header['NORMRR'],tpcfleg.fillHDU(h0['TPCFLegendre']).header['NORMDR']
            nDD = tpcfleg.fillHDU(h0['TPCFLegendre']).header['NORMDD']
            tpcf_mu_int,tpcf_unc_grid = np.zeros((len(sCenters),len(muCenters))),np.zeros((len(sCenters),len(muCenters)))
            for k in range(len(tpcfleg.fillHDU(h0['TPCFLegendre']).data)):
                i,j = tpcfleg.fillHDU(h0['TPCFLegendre']).data['iS'][k],tpcfleg.fillHDU(h0['TPCFLegendre']).data['iMu'][k]
                if (rr[k] != 0.):
                    tpcf_mu_int[i,j] = ((dd[k]/nDD)+(rr[k]/nRR)-2*(dr[k]/nDR))/(rr[k]/nRR)
                    tpcf_unc_grid[i,j] = (np.sqrt(dde2[k])/nDD)/(rr[k]/nRR)
            del rr, dr, dd, dde2
            xi_ell0,xi_ell2,xi_ell4 = self.muIntegralcorr(tpcf_mu_int,sCenters,muCenters,muCenters[1]-muCenters[0])
            xi_ell0_unc,xi_ell2_unc,xi_ell4_unc = self.muIntegralerr(tpcf_unc_grid,sCenters,muCenters,muCenters[1]-muCenters[0])
            hduleg_cols = [fits.Column(name='iS', array=grid, format='I')]
            hduleg_cols.append(fits.Column(name='ell0Corr', array=xi_ell0, format='D'))
            hduleg_cols.append(fits.Column(name='ell0CorrErr', array=xi_ell0_unc, format='D'))
            hduleg_cols.append(fits.Column(name='ell2Corr', array=xi_ell2, format='D'))
            hduleg_cols.append(fits.Column(name='ell2CorrErr', array=xi_ell2_unc, format='D'))
            hduleg_cols.append(fits.Column(name='ell4Corr', array=xi_ell4, format='D'))
            hduleg_cols.append(fits.Column(name='ell4CorrErr', array=xi_ell4_unc, format='D'))
            hduleg = fits.BinTableHDU.from_columns(hduleg_cols,name='Expanded TPCF')
            hduleg.header.add_comment("Legendre multipoles of the two-point correlation function for pairs of galaxies"+
                                   " (using LS estimator) by distance S")
            if grid2d == False:
                hduleg.header['cputime'] = cputime
                hduleg.header['walltime'] = walltime
            self.hdus.append(hduleg)
            
            if grid2d == True:
                iSig,iPi = tpcf2d.fillHDU(h0['TPCF2D']).data['iSigma'],tpcf2d.fillHDU(h0['TPCF2D']).data['iPi']
                rr_2d,dr_2d = tpcf2d.fillHDU(h0['TPCF2D']).data['RR'],tpcf2d.fillHDU(h0['TPCF2D']).data['DR'],
                dd_2d,dde2_2d = tpcf2d.fillHDU(h0['TPCF2D']).data['DD'],tpcf2d.fillHDU(h0['TPCF2D']).data['DDe2']
                hdu2d_cols = [fits.Column(name='iSigma', array=iSig, format='I'),
                              fits.Column(name='iPi', array=iPi, format='I'),
                              fits.Column(name='RR', array=rr_2d, format='D'),
                              fits.Column(name='DR', array=dr_2d, format='D'),
                              fits.Column(name='DD', array=dd_2d, format='D'),
                              fits.Column(name='DDe2', array=dde2_2d, format='D')]
                
                nRR_2d,nDR_2d,nDD_2d = self.getInput('fTheta').header['NORM'],self.getInput('gThetaZ').header['NORM'],self.getInput('uThetaZZ').header['NORM']
                hdu2d_cols = self.lsBundle(hdu2d_cols,self.lsEst(rr_2d,dr_2d,dd_2d,dde2_2d,nRR_2d,nDR_2d,nDD_2d))
                hdu2d = fits.BinTableHDU.from_columns(hdu2d_cols,name='TPCF2D')
                hdu2d.header['NORMRR'] = nRR_2d
                hdu2d.header['NORMDR'] = nDR_2d
                hdu2d.header['NORMDD'] = nDD_2d
                hdu2d.header.add_comment("Two-point correlation function for pairs of galaxies"+
                                         " (using LS estimator) by distances Sigma and Pi")
                hdu2d.header['cputime'] = cputime
                hdu2d.header['walltime'] = walltime
                self.hdus.append(hdu2d)
                self.hdus.append(self.binCenters(self.config.binningSigma(), "centerSigma") )
                self.hdus.append(self.binCenters(self.config.binningPi(), "centerPi") )
            self.writeToFile()
        return

    def combineOutputZ(self):
        zFiles = [self.outputFileName.replace(self.config.name,
                                              '_'.join([self.config.name,
                                                        self.config.suffixZ(iZ)]))
                  for iZ in range(len(self.config.binningsZ()))]
        self.combineOutput(zFiles)

    def plot(self,smax):
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        infile = self.outputFileName

        def getData1Ddists(infile,smax):
            dat_1d = fits.open(infile)[3]
            sep1d = fits.open(infile)[2].data['binCenter']
            RRnorm,DRnorm,DDnorm = (dat_1d.data['RR']/dat_1d.header['NORMRR']),(dat_1d.data['DR']/dat_1d.header['NORMDR']),(dat_1d.data['DD']/dat_1d.header['NORMDD'])
            condition = (sep1d <= smax)
            return(sep1d[condition],RRnorm[condition],DRnorm[condition],DDnorm[condition])
        
        def getData1Dxiss(infile,smax):
            dat_1d = fits.open(infile)[3]
            sep1d = fits.open(infile)[2].data['binCenter']
            sep1derr = (sep1d[1]-sep1d[0])/2
            tpcf1d,tpcf1derr = dat_1d.data['LSCorr'],dat_1d.data['LSCorrErr']
            condition = (sep1d <= smax)
            return(sep1d[condition],sep1derr,tpcf1d[condition],tpcf1derr[condition])

        def getDataExpxiss(infile,smax):
            dat_1d = fits.open(infile)[4]
            sep1d = fits.open(infile)[2].data['binCenter']
            sep1derr = (sep1d[1]-sep1d[0])/2
            tpcfell2,tpcfell2err = dat_1d.data['ell2Corr'],dat_1d.data['ell2CorrErr']
            tpcfell4,tpcfell4err = dat_1d.data['ell4Corr'],dat_1d.data['ell4CorrErr']
            condition = (sep1d <= smax)
            return((sep1d[condition],sep1derr,tpcfell2[condition],tpcfell2err[condition]),(sep1d[condition],sep1derr,tpcfell4[condition],tpcfell4err[condition]))

        def plot1Ddist(pltdat1d):
            yRan = np.max((pltdat1d[1],pltdat1d[2],pltdat1d[3]))-np.min((pltdat1d[1],pltdat1d[2],pltdat1d[3]))
            plt.figure()
            plt.grid(True,ls='-.',alpha=.6)
            plt.title(self.config.__class__.__name__+'\n'+'1D Distributions')
            plt.step(pltdat1d[0],pltdat1d[1],color='darkorange',lw=2,label='RR')
            plt.step(pltdat1d[0],pltdat1d[2],color='forestgreen',lw=2,label='DR')
            plt.step(pltdat1d[0],pltdat1d[3],color='royalblue',lw=2,label='DD')
            plt.legend(framealpha = 1, loc = 'upper left')
            plt.xlabel(r'$s$ [$h^{-1}$Mpc]')
            plt.ylabel(r'Prob.')
            plt.xlim(pltdat1d[0][0],pltdat1d[0][-1])
            plt.ylim(np.min((pltdat1d[1],pltdat1d[2],pltdat1d[3]))-0.04*yRan,np.max((pltdat1d[1],pltdat1d[2],pltdat1d[3]))+0.04*yRan)
            pdf.savefig()
            plt.close()

        def plot1Dxiss(pltdat1d):
            yRan = (pltdat1d[2]*pltdat1d[0]**2).max()-(pltdat1d[2]*pltdat1d[0]**2).min()
            plt.figure()
            plt.grid(True,ls='-.',alpha=.6)
            plt.title(self.config.__class__.__name__+'\n'+'1D Correlation')
            plt.errorbar(pltdat1d[0],pltdat1d[2]*pltdat1d[0]**2,xerr=pltdat1d[1],yerr=pltdat1d[3]*pltdat1d[0]**2,
                         ls='',marker='o',mec='darkblue',ecolor='darkblue',mfc='cadetblue',capsize=2.4,lw=1.2,ms=5)    
            plt.xlabel(r'$s$ [$h^{-1}$Mpc]')
            plt.ylabel(r'$\hat \xi(s) \cdot s^2$')
            plt.xlim(pltdat1d[0][0]-2*pltdat1d[1],pltdat1d[0][-1]+2*pltdat1d[1])
            plt.ylim((pltdat1d[2]*pltdat1d[0]**2).min()-0.1*yRan,(pltdat1d[2]*pltdat1d[0]**2).max()+0.1*yRan)
            pdf.savefig()
            plt.close()

        def plotExpell2(pltdat1d):
            yRan = (pltdat1d[2]*pltdat1d[0]**2).max()-(pltdat1d[2]*pltdat1d[0]**2).min()
            plt.figure()
            plt.grid(True,ls='-.',alpha=.6)
            plt.title(self.config.__class__.__name__+'\n'+'Expanded Correlation $\ell=2$')
            plt.errorbar(pltdat1d[0],pltdat1d[2]*pltdat1d[0]**2,xerr=pltdat1d[1],yerr=pltdat1d[3]*pltdat1d[0]**2,
                         ls='',marker='s',mec='maroon',ecolor='maroon',mfc='salmon',capsize=2.4,lw=1.2,ms=5)
            plt.xlabel(r'$s$ [$h^{-1}$Mpc]')
            plt.ylabel(r'$\hat \xi_{2}(s) \cdot s^2$')
            plt.xlim(pltdat1d[0][0]-2*pltdat1d[1],pltdat1d[0][-1]+2*pltdat1d[1])
            plt.ylim((pltdat1d[2]*pltdat1d[0]**2).min()-0.1*yRan,(pltdat1d[2]*pltdat1d[0]**2).max()+0.1*yRan)
            pdf.savefig()
            plt.close()
            
        def plotExpell4(pltdat1d):
            yRan = (pltdat1d[2]*pltdat1d[0]**2).max()-(pltdat1d[2]*pltdat1d[0]**2).min()
            plt.figure()
            plt.grid(True,ls='-.',alpha=.6)
            plt.title(self.config.__class__.__name__+'\n'+'Expanded Correlation $\ell=4$')
            plt.errorbar(pltdat1d[0],pltdat1d[2]*pltdat1d[0]**2,xerr=pltdat1d[1],yerr=pltdat1d[3]*pltdat1d[0]**2,
                         ls='',marker='X',mec='saddlebrown',ecolor='saddlebrown',mfc='y',capsize=2.4,lw=1.2,ms=7) 
            plt.xlabel(r'$s$ [$h^{-1}$Mpc]')
            plt.ylabel(r'$\hat \xi_{4}(s) \cdot s^2$')
            plt.xlim(pltdat1d[0][0]-2*pltdat1d[1],pltdat1d[0][-1]+2*pltdat1d[1])
            plt.ylim((pltdat1d[2]*pltdat1d[0]**2).min()-0.1*yRan,(pltdat1d[2]*pltdat1d[0]**2).max()+0.1*yRan)
            pdf.savefig()
            plt.close()

        pltdat1ddist = getData1Ddists(infile,smax)
        pltdat1dxiss = getData1Dxiss(infile,smax)
        pltdatExpell2 = getDataExpxiss(infile,smax)[0]
        pltdatExpell4 = getDataExpxiss(infile,smax)[1]
        
        with PdfPages(infile.replace('fits','pdf')) as pdf:
            plot1Ddist(pltdat1ddist)
            plot1Dxiss(pltdat1dxiss)
            plotExpell2(pltdatExpell2)
            plotExpell4(pltdatExpell4)
            print('Wrote %s'% pdf._file.fh.name, file=self.out)
            
        return

class AdderTPCF2D(object):
    def __init__(self, tpcf2d=None, shape2D=None):
        self.items = ['RR','DR','DD','DDe2']
        if not tpcf2d: return
        indices = (tpcf2d.data['iSigma'], tpcf2d.data['iPi'])
        for item in self.items:
            setattr(self, item, csr_matrix((tpcf2d.data[item], indices), shape2D))
        return

    def __add__(self, other):
        thesum = AdderTPCF2D()
        for item in self.items:
            setattr(thesum, item, getattr(self,item) + getattr(other, item))
        return thesum

    def fillHDU(self, hdu):
        allnonzero = sum(getattr(self, item) for item in self.items)
        iSigma, iPi = allnonzero.nonzero()
        hdu.data['iSigma'] = iSigma
        hdu.data['iPi'] = iPi
        for item in self.items:
            hdu.data[item] = getattr(self, item)[iSigma, iPi].A1
        return hdu

class AdderTPCFLegendreGrid(object):
    def __init__(self, tpcfleg=None, shape2D=None):
        self.items = ['RR','DR','DD','DDe2']
        if not tpcfleg: return
        indices = (tpcfleg.data['iS'], tpcfleg.data['iMu'])
        for item in self.items:
            setattr(self, item, csr_matrix((tpcfleg.data[item], indices), shape2D))
        return

    def __add__(self, other):
        thesum = AdderTPCFLegendreGrid()
        for item in self.items:
            setattr(thesum, item, getattr(self,item) + getattr(other, item))
        return thesum

    def fillHDU(self, hdu):
        allnonzero = sum(getattr(self, item) for item in self.items)
        iS, iMu = allnonzero.nonzero()
        hdu.data['iS'] = iS
        hdu.data['iMU'] = iMu
        for item in self.items:
            hdu.data[item] = getattr(self, item)[iS, iMu].A1
        return hdu
