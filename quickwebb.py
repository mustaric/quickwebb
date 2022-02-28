#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 09:29:05 2022

@author: smullally
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from astropy.io import fits


def webbcrawl(path):

    hdr_keywords = ['INSTRUME','MODULE','DETECTOR','FILTER','PUPIL','EFFINTTM','SUBARRAY','TARGNAME','PROGRAM']
    hdr_data = []

    for root, dirs, files in os.walk(path):
        for ff in files:
            if ff[-4:]=='fits':
                ffpath = os.path.join(root,ff)
                hdr = fits.getheader(ffpath)
                hdr_dict = {}

                for keyword in hdr_keywords:
                    if keyword in hdr.keys():
                        hdr_dict[keyword] = hdr[keyword]
                    else:
                        hdr_dict[keyword] = 'none'
                    

                hdr_dict['PATH'] = ffpath
                hdr_data.append(hdr_dict)
           
        
    df = pd.DataFrame.from_dict(hdr_data,orient='columns')
    #df.info_columns=['INSTRUME','MODULE','DETECTOR'.'FILTER','TARGET','PROGRAM']

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', 15)
    #pd.set_option('display.max_columns', 7)
    
    print(df[hdr_keywords])
    
    return(df)

from matplotlib.patches import Circle
from astropy.stats import sigma_clipped_stats
from photutils import aperture as ap
from photutils import centroids

def photnow (ffpath, pos=None, radii_pixels=12, annulus_pixels = 2):
    try:
        data = fits.getdata(ffpath)
        if len(np.shape(data)) != 2:
            print("Data is not an 2D image, fingers crossed this works.")
    except FileNotFoundError:
        print("File Not Found.")
        print(ffpath)
        return
    
    if pos is None:
        print("No (x,y) given, using the center of the image.")
        pos = (int(np.floor(np.shape(data)[0]/2)), 
               int(np.floor(np.shape(data)[1]/2))) 
        print(pos)                
        
    nr = int(radii_pixels)
    aw = int(annulus_pixels)
    pos = (int(pos[0]),int(pos[1]))
    phot_vals = np.zeros(nr)
    radii = np.linspace(1, nr, nr)

    cutout = data[pos[1]-2*nr:pos[1]+2*nr,pos[0]-2*nr:pos[0]+2*nr]
    mini_cut = cutout[nr:3*nr,nr:3*nr]

    cen = centroids.centroid_com(mini_cut) + np.array([nr,nr])

    annulus = ap.CircularAnnulus(cen, r_in=nr, r_out=nr+aw)
    annulus_mask = annulus.to_mask(method='center')
    annulus_data = annulus_mask.multiply(cutout)
    mask = annulus_mask.data    
    annulus_data_1d = annulus_data[mask > 0]
    mean, median, stddev = sigma_clipped_stats(annulus_data_1d)
 
    for ii,radius in enumerate(radii):
        aperture = ap.CircularAperture(cen, r=radii[ii])
        phot_table = ap.aperture_photometry(cutout, [aperture, annulus])   
        phot_val = phot_table['aperture_sum_0'][0] - \
                   phot_table['aperture_sum_1'][0]*aperture.area/annulus.area
        phot_vals[ii] = phot_val

    error = stddev*np.sqrt(aperture.area)    
    
    plotphot(radii, phot_vals, cutout, cen, radii_pixels, annulus_pixels)

    print('Aper. Photometry: %9.3f +/- %9.3f' % (phot_vals[-1], error))
    
    return radii, phot_vals, cutout, cen

def plotphot(radii, phot_vals, cutout, cen, nr, aw):

    vmax = np.percentile(cutout, 99)
    vmin = np.percentile(cutout, 0.5)
    fig = plt.figure(figsize=(6,3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(cutout,cmap='cividis',extent=[-2*nr,2*nr,-2*nr,2*nr], 
               vmin=vmin, vmax=vmax)
    ax1.set_xlabel('Pixels')
    ax1.set_ylabel('Pixels')

    ap_patch = Circle(cen-2*nr,nr, fill=False, color='white')
    an_patch = Circle(cen-2*nr,nr+aw, fill=False, color='yellow')
    ax1.add_patch(ap_patch)
    ax1.add_patch(an_patch)

    ax2.step(radii, phot_vals)
    ax2.set_xlabel('Pixels')

    plt.tight_layout()
    plt.show()