#!/usr/bin/env python

import os
import urllib.request
import argparse

parser = argparse.ArgumentParser(description="This script creates the ../data/ "+\
                                 "directory if it does not already exist."+\
                                 " It then fills the directory with publicly available data.")
parser.add_argument("--dr9", "--DR9", action='store_true', help="Download the SDSS CMASS DR9 catalogs")
parser.add_argument("--dr12", "--DR12", action='store_true', help="Download the SDSS CMASS DR12 catalogs")
args   = parser.parse_args()

out = '../data/'

sdss9_files = ['galaxies_DR9_CMASS_North.fits',
         'galaxies_DR9_CMASS_South.fits',
         'randoms_DR9_CMASS_North.fits',
         'randoms_DR9_CMASS_South.fits']
url9 = "https://data.sdss.org/sas/dr9/boss/lss/"

sdss12_files = ['galaxy_DR12v5_CMASS_North.fits.gz',
         'galaxy_DR12v5_CMASS_South.fits.gz',
         'random0_DR12v5_CMASS_North.fits.gz',
         'random0_DR12v5_CMASS_South.fits.gz']
url12 = "https://data.sdss.org/sas/dr12/boss/lss/"

if not os.path.exists(out):
    os.makedirs(out)

if args.dr9:
    if not os.path.exists(out+"sdss/"):
        os.makedirs(out+"sdss/")
    for f in sdss9_files:
        print()
        print('Downloading', url9+f)
        urllib.request.urlretrieve(url9+f, out+"sdss/"+f)
        print("to", out+"sdss/"+f)

if args.dr12:
    if not os.path.exists(out+"sdss/"):
        os.makedirs(out+"sdss/")
    print()
    print('Caution: This may take some time...')
    for f in sdss12_files:
        print()
        print('Downloading', url12+f)
        urllib.request.urlretrieve(url12+f, out+"sdss/"+f)
        print("to", out+"sdss/"+f)
    print()
    print("To unzip DR12 files, run the following lines")
    print()
    print("gzip ../data/sdss/galaxy_DR12v5_CMASS_North.fits -d")
    print("gzip ../data/sdss/galaxy_DR12v5_CMASS_South.fits -d")
    print("gzip ../data/sdss/random0_DR12v5_CMASS_North.fits -d")
    print("gzip ../data/sdss/random0_DR12v5_CMASS_South.fits -d")
    print()
