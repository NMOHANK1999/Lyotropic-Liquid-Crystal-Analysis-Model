# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 10:49:10 2021

@author: Nishanth
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import pandas as pd
from plantcv import plantcv as pcv
import math
"""
1=DNA
2=PLL
m=droplet mask
cm= coarcervate mask
b=barcode
r=reporter
or=outside reporter
"""
m = cv2.imread(r"C:\Users\Nishanth\Desktop\coarcervate new data\droplets_mask.tif", 0)
cm1 = cv2.imread(r"C:\Users\Nishanth\Desktop\coarcervate new data\ch05_DNA_reporter_coacevate_mask.tif", 0)
cm2 = cv2.imread(r"C:\Users\Nishanth\Desktop\coarcervate new data\ch00_PLL_reporter_coacevate_mask.tif", 0)
b1 = cv2.imread(r"C:\Users\Nishanth\Desktop\coarcervate new data\ch03.tif", 0)
b2 = cv2.imread(r"C:\Users\Nishanth\Desktop\coarcervate new data\ch01_PLL_barcode.tif", 0)
r1 = cv2.imread(r"C:\Users\Nishanth\Desktop\coarcervate new data\ch04_DNA_reporter_coacevate.tif", 0)
r2 = cv2.imread(r"C:\Users\Nishanth\Desktop\coarcervate new data\ch06_PLL_reporter_coacevate.tif", 0)
or1 = cv2.imread(r"C:\Users\Nishanth\Desktop\coarcervate new data\ch05_DNA_reporter_droplet_concentration.tif", 0)
or2 = cv2.imread(r"C:\Users\Nishanth\Desktop\coarcervate new data\ch00_PLL_reporter_droplet_concentration.tif", 0)

df_final= pd.DataFrame()
df_final["Label number"]=float('nan')
df_final["Droplet_Dia"]=float('nan')
df_final["Coarcervate_Dia"]=float('nan')
df_final["Vol_droplet"]=float('nan')
df_final["Vol_CV"]=float('nan')
df_final["Vol_out"]=float('nan')
df_final["VolRatio_CV/Whole"]=float('nan')
df_final["VolRatio_CV/Out"]=float('nan')
df_final["F_bar_DNA"]=float('nan')
df_final["F_bar_PLL"]=float('nan')
df_final["F_CV_DNA"]=float('nan')
df_final["F_CV_PLL"]=float('nan')
df_final["F_out_DNA"]=float('nan')
df_final["F_out_PLL"]=float('nan')
df_final["k_DNA"]=float('nan')
df_final["k_PLL"]=float('nan')
df_final["C_bar_DNA"]=float('nan')
df_final["C_bar_PLL"]=float('nan')
df_final["C_CV_DNA"]=float('nan')
df_final["C_CV_PLL"]=float('nan')
df_final["C_out_DNA"]=float('nan')
df_final["C_out_PLL"]=float('nan')


m_inv= cv2.bitwise_not(m)
cm1_inv= cv2.bitwise_not(cm1)
cm2_inv= cv2.bitwise_not(cm2)

kernal = np.ones((2,2),np.uint8)

m_inv= cv2.erode(m_inv, kernal, iterations=1)



#GENERATING DROPLET MASK

labeled_m_inv= measure.label(m_inv)
props_m_inv = measure.regionprops_table(labeled_m_inv, properties=['label','area','eccentricity'])
df = pd.DataFrame(props_m_inv)
#plt.hist(df['area'], bins=300, range=(0,3000))

lab_m_inv = props_m_inv['label']
ecc_m_inv = props_m_inv['eccentricity']
size_m_inv = props_m_inv['area']
nb_comp_m_inv = lab_m_inv.max()
min_size = 1800
max_size = 2500
e=0.5
mask_droplet = np. zeros(( m_inv.shape ))
for i in range(0, nb_comp_m_inv):
        if (size_m_inv[i] >= min_size and size_m_inv[i] <= max_size and ecc_m_inv[i]<=e):
            mask_droplet[labeled_m_inv == i + 1] = 255

#mask_droplet=pcv.fill_holes(mask_droplet)

mask_droplet = mask_droplet.astype( 'uint8' )


#GENERATING DNA OUTSIDE MASK

dna_out_mask= cv2.bitwise_and(mask_droplet , cm1)

#GENERATING PLL OUTSIDE MASK

pll_out_mask= cv2.bitwise_and(mask_droplet , cm2)



#1droplet
labeled_drop= measure.label(mask_droplet) 
props_drop = measure.regionprops_table(labeled_drop , properties=['label','equivalent_diameter','bbox'])
df_drop = pd.DataFrame(props_drop)
df_final["Label number"] = df_drop['label']
df_final["Droplet_Dia"] =  df_drop['equivalent_diameter']
df_final["Vol_droplet"] =  4/3*3.14* ((df_drop['equivalent_diameter']/2)**3)


#2 barcode

props_b_1 = measure.regionprops_table(labeled_drop, b1, properties=['label', 'mean_intensity','bbox'])

df_b_1 = pd.DataFrame(props_b_1)


#3
 
props_b_2 = measure.regionprops_table(labeled_drop, b2, properties=['label', 'mean_intensity','bbox'])

df_b_2 = pd.DataFrame(props_b_2)


#4 coacervate
labeled_c_1= measure.label(cm1_inv) 
props_c_1 = measure.regionprops_table(labeled_c_1, r1, properties=['label', 'mean_intensity','equivalent_diameter','bbox'])
df_c_1 = pd.DataFrame(props_c_1)

#5
labeled_c_2= measure.label(cm2_inv) 
props_c_2 = measure.regionprops_table(labeled_c_2, r2, properties=['label', 'mean_intensity','equivalent_diameter','bbox'])
df_c_2 = pd.DataFrame(props_c_2)

#6 outside
labeled_out_1= measure.label(dna_out_mask) 
props_out_1 = measure.regionprops_table(labeled_out_1, or1, properties=['label', 'mean_intensity','bbox'])
df_out_1 = pd.DataFrame(props_out_1)

#7
labeled_out_2= measure.label(pll_out_mask) 
props_out_2 = measure.regionprops_table(labeled_out_2, or2, properties=['label', 'mean_intensity','bbox'])
df_out_2 = pd.DataFrame(props_out_2)

#barcode dna
for j in (df_drop.index):
    for i in (df_b_1.index):
        if (df_b_1['bbox-0'][i] >= df_drop['bbox-0'][j] and df_b_1['bbox-1'][i] >= df_drop['bbox-1'][j] and df_b_1['bbox-2'][i] <= df_drop['bbox-2'][j] and df_b_1['bbox-3'][i] <= df_drop['bbox-3'][j]):
            df_final['F_bar_DNA'][j]=df_b_1['mean_intensity'][i]
            
            break

#barcode pll
for j in (df_drop.index):
    for i in (df_b_2.index):
        if (df_b_2['bbox-0'][i] >= df_drop['bbox-0'][j] and df_b_2['bbox-1'][i] >= df_drop['bbox-1'][j] and df_b_2['bbox-2'][i] <= df_drop['bbox-2'][j] and df_b_2['bbox-3'][i] <= df_drop['bbox-3'][j]):
            df_final['F_bar_PLL'][j]=df_b_2['mean_intensity'][i]
            break
    
    
# cv dna
for j in (df_drop.index):
    for i in (df_c_1.index):
        if (df_c_1['bbox-0'][i] >= df_drop['bbox-0'][j] and df_c_1['bbox-1'][i] >= df_drop['bbox-1'][j] and df_c_1['bbox-2'][i] <= df_drop['bbox-2'][j] and df_c_1['bbox-3'][i] <= df_drop['bbox-3'][j]):
            df_final['Coarcervate_Dia'][j]=df_c_1['equivalent_diameter'][i]
            df_final['Vol_CV'][j]=4/3*3.14* ((df_c_1['equivalent_diameter'][i]/2)**3)
            
            df_final['F_CV_DNA'][j]=df_c_1['mean_intensity'][i]
            break
    
# cv pll
for j in (df_drop.index):
    for i in (df_c_2.index):
        if (df_c_2['bbox-0'][i] >= df_drop['bbox-0'][j] and df_c_2['bbox-1'][i] >= df_drop['bbox-1'][j] and df_c_2['bbox-2'][i] <= df_drop['bbox-2'][j] and df_c_2['bbox-3'][i] <= df_drop['bbox-3'][j]):
            
            if (df_final['Coarcervate_Dia'][j] == 0):
                df_final['Coarcervate_Dia'][j]= df_c_2['equivalent_diameter'][i]
                df_final['Vol_CV'][j]= 4/3*3.14* ((df_c_2['equivalent_diameter'][i]/2)**3)
            
            elif (df_final['Coarcervate_Dia'][j] != 0 and df_c_2['equivalent_diameter'][i] != 0):
                df_final['Coarcervate_Dia'][j]= (df_final['Coarcervate_Dia'][j]+df_c_2['equivalent_diameter'][i])/2
                df_final['Vol_CV'][j]= (df_final['Vol_CV'][j] + 4/3*3.14* ((df_c_2['equivalent_diameter'][i]/2)**3))/2
            
            df_final['F_CV_PLL'][j]=df_c_2['mean_intensity'][i]
            break

      
#out dna
for j in (df_drop.index):
    for i in (df_out_1.index):
        if (df_out_1['bbox-0'][i] >= df_drop['bbox-0'][j] and df_out_1['bbox-1'][i] >= df_drop['bbox-1'][j] and df_out_1['bbox-2'][i] <= df_drop['bbox-2'][j] and df_out_1['bbox-3'][i] <= df_drop['bbox-3'][j]):
            
            df_final['F_out_DNA'][j]=df_out_1['mean_intensity'][i]
            break
    
        
# out PLL
for j in (df_drop.index):
    for i in (df_out_2.index):
        if (df_out_2['bbox-0'][i] >= df_drop['bbox-0'][j] and df_out_2['bbox-1'][i] >= df_drop['bbox-1'][j] and df_out_2['bbox-2'][i] <= df_drop['bbox-2'][j] and df_out_2['bbox-3'][i] <= df_drop['bbox-3'][j]):
           
            df_final['F_out_PLL'][j]=df_out_2['mean_intensity'][i]
            break
        

df_final['Vol_out'] = df_final['Droplet_Dia'] - df_final['Coarcervate_Dia']
df_final['VolRatio_CV/Whole'] = df_final['Vol_CV'] / df_final['Vol_droplet']
df_final['VolRatio_CV/Out'] = df_final['Vol_CV'] / df_final['Vol_out']
df_final['k_DNA']=df_final['F_out_DNA'] / df_final['F_CV_DNA']
df_final['k_PLL']=df_final['F_out_PLL'] / df_final['F_CV_PLL']

min1=np.percentile(df_final['F_bar_DNA'], 1)
max1=np.percentile(df_final['F_bar_DNA'], 99)
df_final['C_bar_DNA']= (df_final['F_bar_DNA']-min1)/(max1-min1)

min2=np.percentile(df_final['F_bar_PLL'], 1)
max2=np.percentile(df_final['F_bar_PLL'], 99)
df_final['C_bar_PLL']= (df_final['F_bar_PLL']-min2)/(max2-min2)

#df_final['C_CV_DNA']=0.0
#df_final['C_CV_PLL']=0.0
for a in (df_final.index):
    if (df_final['k_DNA'][a]!=float('nan')):
        df_final['C_CV_DNA'][a]=(df_final['C_bar_DNA'][a]) / (df_final['VolRatio_CV/Whole'][a] + df_final['k_DNA'][a] * (1-df_final['VolRatio_CV/Whole'][a]))
    if(df_final['k_PLL'][a]!=float('nan')): 
         df_final['C_CV_PLL'][a]=(df_final['C_bar_PLL'][a]) / (df_final['VolRatio_CV/Whole'][a] + df_final['k_PLL'][a] * (1-df_final['VolRatio_CV/Whole'][a]))


df_final['C_out_DNA']=df_final['C_CV_DNA'] * df_final['k_DNA']
df_final['C_out_PLL']=df_final['C_CV_PLL'] * df_final['k_PLL']

df_final.to_csv(r"C:\Users\Nishanth\Desktop\drying droplet excel results\new coarcervate data.csv")

# cv2.imwrite(r"C:\Users\Nishanth\Desktop\New folder (6)\mask_droplet2.png", mask_droplet)

# cv2.imshow('Original Image', dna_out_mask)
# cv2.imshow('Original Image2', pll_out_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()