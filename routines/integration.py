from __future__ import print_function
import math
import numpy as np
import lasspia as La
from astropy.io import fits
from scipy.integrate import quad
from scipy.sparse import csr_matrix
from lasspia.timing import timedHDU

class integration(La.routine):

    def __call__(self,grid2d = False):
        if self.nJobs != None:
            self.hdus.append(self.integrationParameters())
            self.hdus.append(self.binCenters(self.config.binningS(), "centerS") )
            self.hdus.extend(self.tpcf())
            self.hdus.append(self.binCenters(self.config.binningSigma(), "centerSigma") )
            self.hdus.append(self.binCenters(self.config.binningPi(), "centerPi") )
            self.writeToFile()
        else:
            try:
                self.hdus.append(self.integrationParameters())
                self.hdus.append(self.binCenters(self.config.binningS(), "centerS") )
                if grid2d == True:
                    self.hdus.extend(self.tpcf(grid2d = True)) 
                    self.hdus.append(self.binCenters(self.config.binningSigma(), "centerSigma") )
                    self.hdus.append(self.binCenters(self.config.binningPi(), "centerPi") )
                else:
                    self.hdus.extend(self.tpcf(grid2d = False))
                self.writeToFile()
            except MemoryError as e:
                print(e.__class__.__name__, e, file=self.out)
                print('\n'.join(['Use less memory by integrating via multiple jobs.',
                                 'For example, use options: --nJobs 8 --nCores 1',
                                 'Then combine job outputs: --nJobs 8',
                                 'Note: This will perform the Legendre expansion on the sigma/pi grid.']),
                                file=self.out)                                                       
        return

    def omegasMKL(self): return self.config.omegasMKL()
    def H0(self): return self.config.H0()
    def muBinEdges(self): return np.linspace(0,1,121)
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

    @timedHDU
    def tpcf(self,grid2d = False):
        self.pdfz = self.getInput('pdfZ').data['probability']
        self.zMask = np.ones(len(self.pdfz)**2, dtype=np.int).reshape(len(self.pdfz),len(self.pdfz))
        self.zMask[:self.config.nBinsMaskZ(),:self.config.nBinsMaskZ()] = 0

        slcT =( slice(None) if self.iJob is None else
                La.slicing.slices(len(self.getInput('centertheta').data),
                                  N=self.nJobs)[self.iJob] )
        def muIntegralcorr(tpcf_mu_int,sCenters,muCenters,dmu):
            xi_ell0 = np.array([2*(tpcf_mu_int[i,:]*dmu).sum()*((2*0+1)/(2)) for i in range(len(sCenters))])
            xi_ell2 =[2*(tpcf_mu_int[i,:]*self.Pell2(muCenters)*dmu).sum()*((2*2+1)/(2)) for i in range(len(sCenters))]
            xi_ell4 = [2*(tpcf_mu_int[i,:]*self.Pell4(muCenters)*dmu).sum()*((2*4+1)/(2)) for i in range(len(sCenters))]
            return(xi_ell0,xi_ell2,xi_ell4)
        
        def muIntegralerr(tpcf_unc_grid,sCenters,muCenters,dmu):            
            xi_ell0_unc = [2*np.sqrt(((tpcf_unc_grid[i,:]*dmu)**2).sum())*((2*0+1)/(2)) for i in range(len(sCenters))]
            xi_ell2_unc =[2*np.sqrt(((tpcf_unc_grid[i,:]*self.Pell2(muCenters)*dmu)**2).sum())*((2*2+1)/(2)) for i in range(len(sCenters))]
            xi_ell4_unc = [2*np.sqrt(((tpcf_unc_grid[i,:]*self.Pell4(muCenters)*dmu)**2).sum())*((2*4+1)/(2)) for i in range(len(sCenters))]
            return(xi_ell0_unc,xi_ell2_unc,xi_ell4_unc)

        def bundleHDU(name, addresses, binning, axes, dropZeros=False, legendre=True, lstep = 0):
            if (legendre == False) & (lstep == 0):
                rr, dr, dd, dde2 = self.calc(addresses, binning, slcT)
                mask = np.logical_or.reduce([a!=0 for a in [rr, dr, dd]]) if dropZeros else np.full(rr.shape, True, dtype=bool)
                idx_adjust = len(self.returnBins(self.config.binningS(), "centerS"))
                grid = []
                for k,iK in zip(axes, np.where(mask)):
                    if len(axes) >1:
                        grid.append(fits.Column(name="i"+k, array=iK-idx_adjust, format='I'))
                    else:
                        grid.append(fits.Column(name="i"+k, array=iK, format='I'))
                        
                hdu = fits.BinTableHDU.from_columns(grid + [
                    fits.Column(name='RR', array=rr[mask], format='D'),
                    fits.Column(name='DR', array=dr[mask], format='D'),
                    fits.Column(name='DD', array=dd[mask], format='D'),
                    fits.Column(name='DDe2', array=dde2[mask], format='D')],
                                                    name=name)

                hdu.header['NORMRR'] = self.getInput('fTheta').header['NORM']
                hdu.header['NORMDR'] = self.getInput('gThetaZ').header['NORM']
                hdu.header['NORMDD'] = self.getInput('uThetaZZ').header['NORM']
                hdu.header.add_comment("Two-point correlation function for pairs of galaxies,"+
                                       " by distance" + ("s" if len(axes)>1 else "") + " " +
                                       " and ".join(axes))
                return hdu
            
            if (legendre == True)&(lstep == 0):
                rr, dr, dd, dde2 = self.calc(addresses, binning, slcT)
                mask = np.logical_or.reduce([a!=0 for a in [rr, dr, dd]]) if dropZeros else np.full(rr.shape, True, dtype=bool)

                grid = [fits.Column(name="i"+k, array=iK, format='I') for k,iK in zip(axes, np.where(mask))]

                columns = grid + [fits.Column(name='RR', array=rr[mask], format='D'),
                                  fits.Column(name='DR', array=dr[mask], format='D'),
                                  fits.Column(name='DD', array=dd[mask], format='D'),
                                  fits.Column(name='DDe2', array=dde2[mask], format='D')]
                tpcf_calc = np.zeros(len(mask))
                tpcf_unc_calc = np.zeros(len(mask))
                for ind in range(len(mask)):
                    if rr[mask][ind] != 0.:
                        tpcf_calc[ind] = ((dd[mask][ind]/self.getInput('uThetaZZ').header['NORM'])+
                                          (rr[mask][ind]/self.getInput('fTheta').header['NORM'])-
                                          2*(dr[mask][ind]/self.getInput('gThetaZ').header['NORM']))/(rr[mask][ind]/self.getInput('fTheta').header['NORM'])
                        tpcf_unc_calc[ind] = (np.sqrt(dde2[mask][ind])/self.getInput('uThetaZZ').header['NORM'])/(rr[mask][ind]/self.getInput('fTheta').header['NORM'])
                columns.append(fits.Column(name='LScorr', array=tpcf_calc, format='D'))
                columns.append(fits.Column(name='LScorrerr', array=tpcf_unc_calc, format='D'))
                
                hdu = fits.BinTableHDU.from_columns(columns,name=name)
                hdu.header['NORMRR'] = self.getInput('fTheta').header['NORM']
                hdu.header['NORMDR'] = self.getInput('gThetaZ').header['NORM']
                hdu.header['NORMDD'] = self.getInput('uThetaZZ').header['NORM']
                hdu.header.add_comment("Two-point correlation function for pairs of galaxies,"+
                                       " by distance" + ("s" if len(axes)>1 else "") + " " +
                                       " and ".join(axes))
                return hdu
            if (legendre == True)&(lstep == 1):
                rr, dr, dd, dde2 = self.calc(addresses, binning, slcT)
                muCenters = (self.muBinEdges()[1:]+self.muBinEdges()[:-1])/2
                sCenters = self.returnBins(self.config.binningS(), "centerS")
                tpcf_mu_int,tpcf_unc_grid = np.zeros((len(sCenters),len(muCenters))),np.zeros((len(sCenters),len(muCenters)))
                nRR_2d,nDR_2d,nDD_2d = self.getInput('fTheta').header['NORM'],self.getInput('gThetaZ').header['NORM'],self.getInput('uThetaZZ').header['NORM']
                for i in range(len(sCenters)):
                    for j in range(len(muCenters)):
                        if (rr[i,j] != 0.):
                           tpcf_mu_int[i,j] = ((dd[i,j]/nDD_2d)+(rr[i,j]/nRR_2d)-2*(dr[i,j]/nDR_2d))/(rr[i,j]/nRR_2d)
                           tpcf_unc_grid[i,j] = (np.sqrt(dde2[i,j])/nDD_2d)/(rr[i,j]/nRR_2d)
                del rr, dr, dd, dde2
                xi_ell0,xi_ell2,xi_ell4 = muIntegralcorr(tpcf_mu_int,sCenters,muCenters,muCenters[1]-muCenters[0])
                xi_ell0_unc,xi_ell2_unc,xi_ell4_unc = muIntegralerr(tpcf_unc_grid,sCenters,muCenters,muCenters[1]-muCenters[0])
                grid = np.asarray(list(range(len(self.returnBins(self.config.binningS(), "centerS")) )))
                columns = [fits.Column(name='iS', array=grid, format='I')]
                columns.append(fits.Column(name='ell0corr', array=xi_ell0, format='D'))
                columns.append(fits.Column(name='ell0correrr', array=xi_ell0_unc, format='D'))
                columns.append(fits.Column(name='ell2corr', array=xi_ell2, format='D'))
                columns.append(fits.Column(name='ell2correrr', array=xi_ell2_unc, format='D'))
                columns.append(fits.Column(name='ell4corr', array=xi_ell4, format='D'))
                columns.append(fits.Column(name='ell4correrr', array=xi_ell4_unc, format='D'))
                hdu = fits.BinTableHDU.from_columns(columns,name='Expanded TPCF')
                hdu.header.add_comment("Legendre multipoles of the TPCF (using LS estimator)")
                return hdu

        if self.nJobs != None:
            sigmaPis = self.sigmaPiGrid(slcT)
            s = np.sqrt(np.power(sigmaPis,2).sum(axis=-1))

            b = self.config.binningDD([self.config.binningS()])
            b2 = self.config.binningDD([self.config.binningSigma(),
                                        self.config.binningPi()])

            hdu = bundleHDU("TPCF", s, b, ["S"],legendre=False,lstep = 0)
            hdu2 = bundleHDU("TPCF2D", sigmaPis, b2, ["Sigma", "Pi"], dropZeros=True,legendre=False,lstep = 0)

            return [hdu2, hdu]
        
        else:                                                                             
            sigmaPis = self.sigmaPiGrid(slcT)                                        
            s = np.sqrt(np.power(sigmaPis,2).sum(axis=-1))                               

            b = self.config.binningDD([self.config.binningS()])                                 
            b2 = self.config.binningDD([self.config.binningSigma(),
                                        self.config.binningPi()])                       

            hdu = bundleHDU("TPCF", s, b, ["S"],legendre=True,lstep = 0)
            hdu2 = bundleHDU("TPCF2D", sigmaPis, b2, ["Sigma", "Pi"], dropZeros=True,legendre=True,lstep = 1)
            if grid2d == True: 
                hdu3 = bundleHDU("TPCF2D", sigmaPis, b2, ["Sigma", "Pi"], dropZeros=True,legendre=False,lstep = 0)
                return [hdu,hdu2,hdu3]
            else: return [hdu,hdu2]
            
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

    def calc(self, *args):
        return (self.calcRR(*args),
                self.calcDR(*args),
                self.calcDD(*args, wName='count'),
                self.calcDD(*args, wName='err2'))

    def calcRR(self, addresses, binning, slcT):
        if self.nJobs != None:
            ft = self.getInput('fTheta').data['count'][slcT]
            counts = ft[:,None,None] * self.pdfz[None,:,None] * self.pdfz[None,None,:] * self.zMask[None,:]
            N = counts.size
            D = addresses.size // N
            rr = np.histogramdd(addresses.reshape(N,D), weights=counts.reshape(N), **binning)[0]
            del counts
            return rr
        else:
            ft = self.getInput('fTheta').data['count'][slcT]
            counts = ft[:,None,None] * self.pdfz[None,:,None] * self.pdfz[None,None,:] * self.zMask[None,:]
            N = counts.size
            D = addresses.size // N
            if D == 1:
                rr = np.histogramdd(addresses.reshape(N,D), weights=counts.reshape(N), **binning)[0]
                del counts
                return rr
            if D == 2:
                muBinEdges = self.muBinEdges()
                halfspace = (self.returnBins(self.config.binningS(), "centerS")[1]-self.returnBins(self.config.binningS(), "centerS")[0])/2
                sBinEdges = np.linspace(self.returnBins(self.config.binningS(), "centerS")[0]-halfspace,
                                        self.returnBins(self.config.binningS(), "centerS")[-1]+halfspace,
                                        len(self.returnBins(self.config.binningS(), "centerS"))+1)
                spairs = (np.sqrt((addresses.reshape(N,D)[:,0])**2+(addresses.reshape(N,D)[:,1])**2))
                mupairs = addresses.reshape(N,D)[:,1]/spairs
                rr = np.histogram2d(spairs,mupairs, weights=counts.reshape(N),bins=[sBinEdges,muBinEdges])[0]
                del counts
                return rr

    def calcDR(self, addresses, binning, slcT):
        if self.nJobs != None:
            gtz = self.getInput('gThetaZ').data
            counts = gtz[slcT,:,None] * self.pdfz[None,None,:] * self.zMask[None,:]
            N = counts.size
            D = addresses.size // N
            dr = np.histogramdd(addresses.reshape(N,D), weights=counts.reshape(N), **binning)[0]
            del counts
            return dr
        else:
            gtz = self.getInput('gThetaZ').data
            counts = gtz[slcT,:,None] * self.pdfz[None,None,:] * self.zMask[None,:]
            N = counts.size
            D = addresses.size // N
            if D == 1:
                dr = np.histogramdd(addresses.reshape(N,D), weights=counts.reshape(N), **binning)[0]
                del counts
                return dr
            if D == 2:
                muBinEdges = self.muBinEdges()
                halfspace = (self.returnBins(self.config.binningS(), "centerS")[1]-self.returnBins(self.config.binningS(), "centerS")[0])/2
                sBinEdges = np.linspace(self.returnBins(self.config.binningS(), "centerS")[0]-halfspace,
                                        self.returnBins(self.config.binningS(), "centerS")[-1]+halfspace,
                                        len(self.returnBins(self.config.binningS(), "centerS"))+1)
                spairs = (np.sqrt((addresses.reshape(N,D)[:,0])**2+(addresses.reshape(N,D)[:,1])**2))
                mupairs = addresses.reshape(N,D)[:,1]/spairs
                dr = np.histogram2d(spairs,mupairs, weights=counts.reshape(N),bins=[sBinEdges,muBinEdges])[0]
                del counts
                return dr

    def calcDD(self, addresses, binning, slcT, wName='count'):
        if self.nJobs != None:
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
        else:
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
            
            if addresses[iTh,iZ,iZ2].shape == (len(addresses[iTh,iZ,iZ2]),2):
                spairs = (np.sqrt((addresses[iTh,iZ,iZ2][:,0])**2+(addresses[iTh,iZ,iZ2][:,1])**2))
                mupairs = addresses[iTh,iZ,iZ2][:,1]/spairs
                muBinEdges = self.muBinEdges()
                halfspace = (self.returnBins(self.config.binningS(), "centerS")[1]-self.returnBins(self.config.binningS(), "centerS")[0])/2
                sBinEdges = np.linspace(self.returnBins(self.config.binningS(), "centerS")[0]-halfspace,
                                        self.returnBins(self.config.binningS(), "centerS")[-1]+halfspace,
                                        len(self.returnBins(self.config.binningS(), "centerS"))+1)
                
                dd = np.histogram2d(spairs,mupairs, weights=counts,bins=[sBinEdges,muBinEdges])[0]
                del counts
                return dd

            else:
                dd = np.histogramdd(addresses[iTh,iZ,iZ2], weights=counts, **binning)[0]
                del counts
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

    def combineOutput(self, jobFiles = None, grid2d = False):
        if not jobFiles:
            jobFiles = [self.outputFileName + self.jobString(iJob)
                        for iJob in range(self.nJobs)]

        shape2D = (self.config.binningSigma()['bins'], self.config.binningPi()['bins'])

        with fits.open(jobFiles[0]) as h0:
            for h in ['parameters','centerS']:
                self.hdus.append(h0[h])
            hdu = h0['TPCF']
            cputime = hdu.header['cputime']
            tpcf2d = AdderTPCF2D(h0['TPCF2D'], shape2D)
            
            for jF in jobFiles[1:]:
                with fits.open(jF) as jfh:
                    assert np.all(h0['parameters'].header[item] == jfh['parameters'].header[item]
                                  for item in ['lightspd','H0','omegaM','omegaK','omegaL'])
                    for axis in ['centerS']:
                        assert np.all(h0[axis].data['binCenter'] == jfh[axis].data['binCenter'])

                    cputime += jfh['TPCF'].header['cputime']
                    tpcf2d += AdderTPCF2D(jfh['TPCF2D'], shape2D)
                    for col in ['RR','DR','DD','DDe2']:
                        hdu.data[col] += jfh['TPCF'].data[col]
                        
            tpcf_calc = np.zeros(len(hdu.data))
            tpcf_unc_calc = np.zeros(len(hdu.data))
            for i in range(len(hdu.data)):
                if hdu.data['RR'][i] !=0:
                    tpcf_calc[i] = ((hdu.data['DD'][i]/hdu.header['NORMDD'])+(hdu.data['RR'][i]/hdu.header['NORMRR'])
                                    -2*(hdu.data['DR'][i]/hdu.header['NORMDR']))/(hdu.data['RR'][i]/hdu.header['NORMRR'])
                    tpcf_unc_calc[i] = (np.sqrt(hdu.data['DDe2'][i])/hdu.header['NORMDD'])/(hdu.data['RR'][i]/hdu.header['NORMRR'])

            new_columns = []
            for ocol in ['iS','RR','DR','DD','DDe2']:
                new_columns.append(fits.Column(name=ocol, array=hdu.data[ocol], format='D'))  
            new_columns.append(fits.Column(name='LScorr', array=tpcf_calc, format='D'))
            new_columns.append(fits.Column(name='LScorrerr', array=tpcf_unc_calc, format='D')) 
            new_1d_hdu = fits.BinTableHDU.from_columns(new_columns)

            for head in ['NORMRR','NORMDR','NORMDD','cputime']:
                new_1d_hdu.header[head] = hdu.header[head]
                
            new_1d_hdu.header['cputime'] = cputime
            self.hdus.append(new_1d_hdu)
            table2d = tpcf2d.fillHDU(h0['TPCF2D'])
            gLen = len(hdu.data['iS'])
            tpcf_sigpi = np.zeros((gLen,gLen))
            tpcferr_sigpi = np.zeros((gLen,gLen))
            MuatLoc = np.zeros((gLen,gLen))
            SatLoc = np.zeros((gLen,gLen))
            sCenters = self.returnBins(self.config.binningS(), "centerS")
            
            for idx in range(len(table2d.data)):
                sepCalc = np.sqrt(sCenters[table2d.data['iPi'][idx]]**2+sCenters[table2d.data['iSigma'][idx]]**2)
                MuatLoc[table2d.data['iPi'][idx],table2d.data['iSigma'][idx]] = sCenters[table2d.data['iPi'][idx]]/sepCalc
                SatLoc[table2d.data['iPi'][idx],table2d.data['iSigma'][idx]] = sepCalc
                if table2d.data['RR'][idx] != 0.:
                    tval = ((table2d.data['DD'][idx]/table2d.header['NORMDD'])+(table2d.data['RR'][idx]/table2d.header['NORMRR'])-
                            2*(table2d.data['DR'][idx]/table2d.header['NORMDR']))/(table2d.data['RR'][idx]/table2d.header['NORMRR'])
                    tuncval = (np.sqrt(table2d.data['DDe2'][idx])/table2d.header['NORMDD'])/(table2d.data['RR'][idx]/table2d.header['NORMRR'])
                    tpcf_sigpi[table2d.data['iPi'][idx],table2d.data['iSigma'][idx]] = tval
                    tpcferr_sigpi[table2d.data['iPi'][idx],table2d.data['iSigma'][idx]] = tuncval

            halfspace = (sCenters[1]-sCenters[0])/2
            sBinEdges = np.linspace(sCenters[0]-halfspace,sCenters[-1]+halfspace,len(sCenters)+1)

            dmuGrid = (2*halfspace)/SatLoc
            xi_ell0_est = np.zeros(gLen)
            xi_ell0err_est = np.zeros(gLen)

            xi_ell2_est = np.zeros(gLen)
            xi_ell2err_est = np.zeros(gLen)

            xi_ell4_est = np.zeros(gLen)
            xi_ell4err_est = np.zeros(gLen)

            for j in range(gLen):
                cond = (SatLoc > sBinEdges[j])&(SatLoc <= sBinEdges[j+1])
                xi_ell0_est[j] = (tpcf_sigpi[cond]*dmuGrid[cond]).sum()
                xi_ell0err_est[j] = (tpcferr_sigpi[cond]*dmuGrid[cond]).sum()
                xi_ell2_est[j] = (2*(2*2+1)/2)*(self.Pell2(MuatLoc[cond])*tpcf_sigpi[cond]*dmuGrid[cond]).sum()
                xi_ell2err_est[j] = (2*(2*2+1)/2)*(self.Pell2(MuatLoc[cond])*tpcferr_sigpi[cond]*dmuGrid[cond]).sum()
                xi_ell4_est[j] = (2*(2*4+1)/2)*(self.Pell4(MuatLoc[cond])*tpcf_sigpi[cond]*dmuGrid[cond]).sum()
                xi_ell4err_est[j] = (2*(2*4+1)/2)*(self.Pell4(MuatLoc[cond])*tpcferr_sigpi[cond]*dmuGrid[cond]).sum()

            grid = np.asarray(list(range(len(self.returnBins(self.config.binningS(), "centerS")) )))
            columns = [fits.Column(name='iS', array=grid, format='I')]
            columns.append(fits.Column(name='ell0corr', array=xi_ell0_est, format='D'))
            columns.append(fits.Column(name='ell0correrr', array=xi_ell0err_est, format='D'))
            columns.append(fits.Column(name='ell2corr', array=xi_ell2_est, format='D'))
            columns.append(fits.Column(name='ell2correrr', array=xi_ell2err_est, format='D'))
            columns.append(fits.Column(name='ell4corr', array=xi_ell4_est, format='D'))
            columns.append(fits.Column(name='ell4correrr', array=xi_ell4err_est, format='D'))
            hduleg = fits.BinTableHDU.from_columns(columns,name='Expanded TPCF')
            hduleg.header.add_comment("Legendre multipoles of the TPCF (using LS estimator)"+"Caution, these multipoles are estimated!")
            self.hdus.append(hduleg)
            
            if grid2d == True:
                self.hdus.append(table2d)
                self.hdus.append(h0[5])            
                self.hdus.append(h0[6])

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
            tpcf1d,tpcf1derr = dat_1d.data['LScorr'],dat_1d.data['LScorrerr']
            condition = (sep1d <= smax)
            return(sep1d[condition],sep1derr,tpcf1d[condition],tpcf1derr[condition])

        def getDataExpxiss(infile,smax):
            dat_1d = fits.open(infile)[4]
            sep1d = fits.open(infile)[2].data['binCenter']
            sep1derr = (sep1d[1]-sep1d[0])/2
            tpcfell2,tpcfell2err = dat_1d.data['ell2corr'],dat_1d.data['ell2correrr']
            tpcfell4,tpcfell4err = dat_1d.data['ell4corr'],dat_1d.data['ell4correrr']
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
                         ls='',marker='o',mfc='grey',mec='k',color='k',capsize=2.4,lw=1.2,ms=5)
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
                         ls='',marker='s',mfc='r',mec='maroon',color='maroon',capsize=2.4,lw=1.2,ms=5)
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
                         ls='',marker='^',mfc='limegreen',mec='darkgreen',color='darkgreen',capsize=2.4,lw=1.2,ms=6)
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
