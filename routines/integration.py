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
                self.hdus.extend(self.tpcf())                                                
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
    def tpcf(self):
        self.pdfz = self.getInput('pdfZ').data['probability']
        self.zMask = np.ones(len(self.pdfz)**2, dtype=np.int).reshape(len(self.pdfz),len(self.pdfz))
        self.zMask[:self.config.nBinsMaskZ(),:self.config.nBinsMaskZ()] = 0

        slcT =( slice(None) if self.iJob is None else
                La.slicing.slices(len(self.getInput('centertheta').data),
                                  N=self.nJobs)[self.iJob] )
        def muIntegralcorr(tpcf_mu_int,sCenters,muCenters,dmu):
            xi_ell0 = np.array([2*(tpcf_mu_int[i,:]*dmu).sum()*((2*0+1)/(2)) for i in range(len(sCenters))])
            xi_ell2 =[2*(tpcf_mu_int[i,:]*self.Pell2(muCenters)*dmu).sum()*((2*0+1)/(2)) for i in range(len(sCenters))]
            xi_ell4 = [2*(tpcf_mu_int[i,:]*self.Pell4(muCenters)*dmu).sum()*((2*0+1)/(2)) for i in range(len(sCenters))]
            return(xi_ell0,xi_ell2,xi_ell4)
        
        def muIntegralerr(tpcf_unc_grid,sCenters,muCenters,dmu):            
            xi_ell0_unc = [2*np.sqrt(((tpcf_unc_grid[i,:]*dmu)**2).sum())*((2*0+1)/(2)) for i in range(len(sCenters))]
            xi_ell2_unc =[2*np.sqrt(((tpcf_unc_grid[i,:]*self.Pell2(muCenters)*dmu)**2).sum())*((2*2+1)/(2)) for i in range(len(sCenters))]
            xi_ell4_unc = [2*np.sqrt(((tpcf_unc_grid[i,:]*self.Pell4(muCenters)*dmu)**2).sum())*((2*2+1)/(2)) for i in range(len(sCenters))]
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
                        tpcf_calc[ind] = ((dd[mask][ind]/self.getInput('uThetaZZ').header['NORM'])+(rr[mask][ind]/self.getInput('fTheta').header['NORM'])-2*(dr[mask][ind]/self.getInput('gThetaZ').header['NORM']))/(rr[mask][ind]/self.getInput('fTheta').header['NORM'])
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
                hdu = fits.BinTableHDU.from_columns(columns,name=name)
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

            return [hdu,hdu2]
            #return [hdu2]                                                                   
            

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
            
            # 2D !
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

            # 1D !
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

    def combineOutput(self, jobFiles = None):
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
            hdu.header['cputime'] = cputime
        
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

            for head in ['NORMDD','NORMDR','NORMDD','cputime']:
                new_1d_hdu.header[head] = hdu.header[head]
            
            self.hdus.append(new_1d_hdu)
            #self.hdus.append(h0[5]) #Sigmas            
            #self.hdus.append(h0[6]) #Pis
            self.writeToFile()
        return

    def combineOutputZ(self):
        zFiles = [self.outputFileName.replace(self.config.name,
                                              '_'.join([self.config.name,
                                                        self.config.suffixZ(iZ)]))
                  for iZ in range(len(self.config.binningsZ()))]
        self.combineOutput(zFiles)

    def plot(self):
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        infile = self.outputFileName

        tpcf = fits.getdata(infile, 'TPCF')
        centerS = fits.getdata(infile, 'centerS').binCenter
        nRR,nDR,nDD = (lambda h:
                       (h['normrr'],
                        h['normdr'],
                        h['normdd']))(fits.getheader(infile, 'TPCF'))

        def tpcfPlot(pdf, binFactor):
            s = centerS[tpcf.iS]
            iStop = len(s) // binFactor
            plt.figure()
            plt.title(self.config.__class__.__name__)
            plt.step(s[:iStop], tpcf.RR[:iStop]/nRR, where='mid', label='RR', linewidth=0.4)
            plt.step(s[:iStop], tpcf.DR[:iStop]/nDR, where='mid', label='DR', linewidth=0.4)
            plt.step(s[:iStop], tpcf.DD[:iStop]/nDD, where='mid', label='DD', linewidth=0.4)
            plt.legend()
            plt.xlabel('s')
            plt.ylabel('probability')
            pdf.savefig()
            plt.close()

        def xissPlot(pdf, sMax):
            S = centerS[tpcf.iS]
            iStop = None if S[-1]<sMax else next(iter(np.where(S >= sMax)[0]))
            s = S[:iStop]
            xi = ( (tpcf.RR[:iStop]/nRR + tpcf.DD[:iStop]/nDD - 2*tpcf.DR[:iStop]/nDR)
                   / (tpcf.RR[:iStop]/nRR) )
            xie = ( (np.sqrt(tpcf.DDe2[:iStop])/nDD)
                    / (tpcf.RR[:iStop]/nRR) )
            ds = s[1]-s[0]

            plt.figure()
            plt.title(self.config.__class__.__name__)
            plt.errorbar(s, (xi*s*s), yerr=(xie*s*s), xerr=ds/2, fmt='.')
            plt.xlabel(r"$\mathrm{s\ [h^{-1} Mpc]}$")
            plt.ylabel(r"$\mathrm{\xi(s)s^2}$")
            plt.grid()
            pdf.savefig()
            plt.close()

        with PdfPages(infile.replace('fits','pdf')) as pdf:
            for i in range(5):
                tpcfPlot(pdf, 2**i)
            xissPlot(pdf, 200)
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
