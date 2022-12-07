import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import pandas as pd
from plantcv import plantcv as pcv
import math

"""
INPUT ALL IMAGES
"""
imgbf = cv2.imread(r"C:\Users\Nishanth\Desktop\20\7bf.tif", 0)
img_dna_barcode = cv2.imread(r"C:\Users\Nishanth\Desktop\20\Dbar7.tif", 0)
img_pep_barcode = cv2.imread(r"C:/Users/Nishanth/Desktop/20/Pbar7.tif", 0)
img_dna_rep = cv2.imread(r"C:\Users\Nishanth\Desktop\20\prep7.tif",0)
#img_pep_rep = cv2.imread(r"C:\Users\Nishanth\Desktop\20\prep1.tif", 0)


#brightfield image

#imgbf= imgbf[424:1624, 424:1624]
#bluring function
blur_img = cv2.GaussianBlur(imgbf, (3,3), 0, borderType=cv2.BORDER_CONSTANT)

#display the blured and original image
#cv2.imshow("Original", img)
#cv2.waitKey(0)          
#cv2.destroyAllWindows()


#creating the theshold of the bright field image
ret, threshbf = cv2.threshold(blur_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#plt.imshow(thresh)

kernal = np.ones((2,2),np.uint8)
threshbf2=pcv.fill_holes(threshbf)
#threshbf3= cv2.erode(threshbf2, kernal, iterations=1)

#cv2.imshow("thresh", thresh)
#cv2.imshow("thresh1", thresh2)
#cv2.imshow("thresh3", thresh3)
#cv2.waitKey(0)          
#cv2.destroyAllWindows()


labeled2_bf= measure.label(threshbf2)

props2_bf = measure.regionprops_table(labeled2_bf, 
                                  properties=['label','area','eccentricity'])
lab2_bf = props2_bf['label']
ecc2_bf = props2_bf['eccentricity']
size2_bf = props2_bf['area']
nb_comp2_bf = lab2_bf.max()
min_size2_bf = 8000
max_size2_bf = 10000
e=1
mask_bf_final = np. zeros(( labeled2_bf.shape ))
for i in range(0, nb_comp2_bf):
        if (size2_bf[i] >= min_size2_bf and size2_bf[i] <= max_size2_bf and ecc2_bf[i]<=e):
            mask_bf_final[labeled2_bf == i + 1] = 255


cv2.imshow("thresh4", mask_bf_final)
cv2.waitKey(0)          
cv2.destroyAllWindows()

# MAKING MASK 1(OF WHOLE DROPLET POSITIONS) 
labeled= measure.label(mask_bf_final)
#plt.imshow(labeled)

#finding properties
props = measure.regionprops_table(labeled, 
                                  properties=['label','area', 'equivalent_diameter', 'bbox'])
#initializing dataframe
df = pd.DataFrame(props)

#ploting histogram to find appropriate test droplet area 
plt.hist(df['area'], bins=300, range=(0,20000))

"""
DNA BARCODE

"""
# Barcode for the DNA 

#img_dna_barcode= img_dna_barcode[424:1624, 424:1624]

img_dna_barcode = cv2.GaussianBlur(img_dna_barcode, (3,3), 0, borderType=cv2.BORDER_CONSTANT)

#FINDING PROPERTIES OF BARCODE
props1 = measure.regionprops_table(labeled, img_dna_barcode, 
                                  properties=['label','area', 'equivalent_diameter', 'mean_intensity'])
#INITIALIZING DATAFRAME 1
df1 = pd.DataFrame(props1)



#finding the normalized conc

"""
#use quantile for finding the min and max
"""

min1=np.percentile(df1['mean_intensity'], 1)
max1=np.percentile(df1['mean_intensity'], 99)
'''
min1=df1['mean_intensity'].min()
max1=df1['mean_intensity'].max()
'''
df1['Norm_conc']= (df1['mean_intensity']-min1)/(max1-min1)

#plt.hist(df1['mean_intensity'], bins=255, range=(0,255))

#cv2.imshow("2",img_dna_barcode )
#cv2.waitKey(0)          
#cv2.destroyAllWindows()

"""
PEPTIDE BARCODE

"""


#img_pep_barcode= img_pep_barcode[424:1624, 424:1624]
img_pep_barcode = cv2.GaussianBlur(img_pep_barcode, (3,3), 0, borderType=cv2.BORDER_CONSTANT)
props2 = measure.regionprops_table(labeled, img_pep_barcode, 
                                  properties=['label','area', 'equivalent_diameter', 'mean_intensity'])
df2 = pd.DataFrame(props2)

#finding the normalized conc
min2=np.percentile(df2['mean_intensity'], 1)
max2=np.percentile(df2['mean_intensity'], 99)
'''
min2=df2['mean_intensity'].min()
max2=df2['mean_intensity'].max()
'''
df2['Norm_conc']= (df2['mean_intensity']-min2)/(max2-min2)


"""

REPORTER OF DNA

"""


#IMPORTING REPORTER IMAGE AND APPLYING BLUR

#img_dna_rep= img_dna_rep[424:1624, 424:1624]

blur_img_dna_rep = cv2.GaussianBlur(img_dna_rep, (3,3), 0, borderType=cv2.BORDER_CONSTANT)
#plt.hist(img_dna_rep.flat, bins=255, range=(0,255))
#FINDING PROPERTIES OF WHOLE DROPLET USING MASK 1
propswhole = measure.regionprops_table(labeled, blur_img_dna_rep , 
                                  properties=['label','area','equivalent_diameter','bbox'])

#INITIALISING DF WHOLE
dfwhole = pd.DataFrame(propswhole)
#print(df3.head())

#FINDING THE VALUE OF THE AREA OF CV USING THE 
#dfwhole["area_of_CV"]= dfwhole["mean_intensity"]*dfwhole["area"]/255.0

#MAKING ANOTHER COLUMN NAMED NUMBER WHICH IS SAME AS ROW INDEX
dfwhole['number']=dfwhole['label']-1


#MAKING MASK 2 (MASK FOR CV) MUST RECALIBRATE FOR OPTIMAL THRESHOLD VALUE

#########################
retrepdna=np.percentile(blur_img_dna_rep.flat, 98)
retrep, threshrep = cv2.threshold(blur_img_dna_rep ,retrepdna,255,cv2.THRESH_BINARY)



thresh2_1=pcv.fill_holes(threshrep)
thresh3_1= cv2.erode(thresh2_1, kernal, iterations=5)

#cv2.imshow("thresh", thresh)
#cv2.imshow("thresh1", thresh2)
cv2.imshow("blur_img4", blur_img_dna_rep )
cv2.imshow("thresh3", thresh3_1)
cv2.waitKey(0)          
cv2.destroyAllWindows()


image_1 = thresh3_1.astype( 'uint8' )

labeled2_dna= measure.label(thresh3_1)

props2_dna = measure.regionprops_table(labeled2_dna, 
                                  properties=['label','area','eccentricity'])
lab2_dna = props2_dna['label']
ecc2_dna = props2_dna['eccentricity']
size2_dna = props2_dna['area']
nb_comp2_dna = lab2_dna.max()
min_size2_dna = 10
max_size2_dna = 900
e=0.8
mask_cv_final = np. zeros(( labeled2_dna.shape ))
for i in range(0, nb_comp2_dna):
        if (size2_dna[i] >= min_size2_dna and size2_dna[i] <= max_size2_dna and ecc2_dna[i]<=e):
            mask_cv_final[labeled2_dna == i + 1] = 255




################

cv2.imshow("blur_img4", blur_img_dna_rep )
cv2.imshow("CV", mask_cv_final)
cv2.waitKey(0)          
cv2.destroyAllWindows()

#LABELING FOR MASK 2
labeledcv= measure.label(mask_cv_final)

#FINDING PROPERTIES OF CV ONLY WITH MASK 2 
propscv = measure.regionprops_table(labeledcv,img_dna_rep, 
                                  properties=['label','area', 'mean_intensity','equivalent_diameter','bbox'])

#INITIALIZING VALUES IN DATAFRAME FOR CV
dfcv = pd.DataFrame(propscv)
dfcv['number']=dfcv['label']-1

#to decide what suitable area we should take the  
plt.hist(dfcv['area'], bins=300, range=(0,2000))

#FINDING INVERSE OF MASK2 AND DISPLAYING IT
maskt = mask_cv_final.astype(np.uint8)
notmask= cv2.bitwise_not(maskt)
#cv2.imshow("not_thresh", notmask)
#cv2.imshow("thresh",mask_cv_final)
#cv2.waitKey(0)          
#cv2.destroyAllWindows()
#cv2.imwrite(r"C:\Users\Nishanth\Desktop\coarcervates\test images\maskcv.jpg", mask_cv_final)

#FINDING MASK 3 OUTSIDE REGION OF COARCERVATE AND DISPLAYING IT

#
maskt1 = mask_bf_final.astype(np.uint8)
new_mask= cv2.bitwise_and(maskt1 , notmask)
cv2.imshow("new_mask", new_mask)
cv2.imshow("blur_img4", blur_img_dna_rep )
cv2.waitKey(0)          
cv2.destroyAllWindows()
#writing 
#cv2.imwrite(r"C:\Users\Nishanth\Desktop\20 test\pep7.jpg", new_mask)

#LABELING THE OUTSIDE REGION
labeled_out= measure.label(new_mask)

#FINDING PROPERTIES OF THE OUTSIDE REGION
props_out= measure.regionprops_table(labeled_out,img_dna_rep ,
                                     properties=['label','area', 'mean_intensity', 'bbox'])

#INITIALIZING THE OUTSIDE DATA FRAME, AND MAKING THE "NUMBER" COLUMN
df_out= pd.DataFrame(props_out) 
df_out['number']=df_out['label']-1

dfwhole['area_of_cv2']=0
dfwhole['equivalent_dia_of_cv']=0
dfwhole['intensity_of_cv']=0
dfwhole['area_of_out']=0
dfwhole['intensity_of_out']=0

# Matching the labels of the CV and the whole DROPLET 
maxcv=dfcv['number'].max()
for j in (dfwhole['number']):
    for i in (dfcv['number']):
        if (dfcv['bbox-0'][i] >= dfwhole['bbox-0'][j] and dfcv['bbox-1'][i] >= dfwhole['bbox-1'][j] and dfcv['bbox-2'][i] <= dfwhole['bbox-2'][j] and dfcv['bbox-3'][i] <= dfwhole['bbox-3'][j]):
            dfwhole['area_of_cv2'][j]=dfcv['area'][i]
            dfwhole['equivalent_dia_of_cv'][j]=dfcv['equivalent_diameter'][i]
            dfwhole['intensity_of_cv'][j]=dfcv['mean_intensity'][i]
            break
     
            
# Matching the labels of outside region of the droplet and whole DROPLET               
maxwhole=dfwhole['number'].max()
maxout=df_out['number'].max()
for i in df_out['number']:
    for j in dfwhole['number']:
        if (df_out['bbox-0'][i] == dfwhole['bbox-0'][j] and df_out['bbox-1'][i] == dfwhole['bbox-1'][j] and df_out['bbox-2'][i] == dfwhole['bbox-2'][j] and df_out['bbox-3'][i] == dfwhole['bbox-3'][j]):
            dfwhole['area_of_out'][j]=df_out['area'][i]          
            dfwhole['intensity_of_out'][j]=df_out['mean_intensity'][i]
            break
            
            
            


#FINDING THE VALUES OF THE VOLUME OF THE WHOLE DROPLET AND THE VOLUME OF THE CV
dfwhole['volume']= 4/3*3.14* (dfwhole['equivalent_diameter']/2)**3
dfwhole['volumeof_cv']= 4/3*3.14* (dfwhole['equivalent_dia_of_cv']/2)**3
dfwhole['volume ratio pep']= dfwhole['volumeof_cv']/dfwhole['volume']

# '''
# REPORTER OF PEPTIDE
# '''


# #img_pep_rep= img_pep_rep[424:1624, 424:1624]

# blur_img_pep_rep = cv2.GaussianBlur(img_pep_rep, (3,3), 0, borderType=cv2.BORDER_CONSTANT)

# #FINDING PROPERTIES OF WHOLE DROPLET USING MASK 1
# propswhole_pep = measure.regionprops_table(labeled, img_pep_rep , 
#                                   properties=['label','area','equivalent_diameter','bbox'])

# #INITIALISING DF WHOLE
# dfwhole_pep = pd.DataFrame(propswhole_pep)

# #FINDING THE VALUE OF THE AREA OF CV USING THE 
# #dfwhole["area_of_CV"]= dfwhole["mean_intensity"]*dfwhole["area"]/255.0

# #MAKING ANOTHER COLUMN NAMED NUMBER WHICH IS SAME AS ROW INDEX
# dfwhole_pep['number']=dfwhole_pep['label']-1


# #MAKING MASK 2 (MASK FOR CV) MUST RECALIBRATE FOR OPTIMAL THRESHOLD VALUE


# #retrep_pep, threshrep_pep = cv2.threshold(blur_img_pep_rep ,50,255,cv2.THRESH_BINARY)
# #retrep_pep, threshrep_pep = cv2.threshold(blur_img_pep_rep ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# retreppep=np.percentile(img_pep_rep.flat, 99)
# retrep_pep, threshrep_pep =cv2.threshold(blur_img_pep_rep ,retreppep,255,cv2.THRESH_BINARY)
# #plt.imshow(threshrep_pep)


# thresh2_1_pep=pcv.fill_holes(threshrep_pep)
# thresh3_1_pep= cv2.erode(thresh2_1_pep, kernal, iterations=2)

# #cv2.imshow("thresh_pep", threshrep_pep)
# cv2.imshow("thresh1", blur_img_pep_rep)
# cv2.imshow("thresh3", thresh3_1_pep)
# cv2.waitKey(0)          
# cv2.destroyAllWindows()


# image_1_pep = thresh3_1_pep.astype( 'uint8' )

# ## This function is a masking fuction with eccentricity less than 0.8
# labeled2_pep= measure.label(thresh3_1_pep)

# props2_pep = measure.regionprops_table(labeled2_pep, 
#                                   properties=['label','area','eccentricity'])
# lab_pep = props2_pep['label']
# ecc_pep = props2_pep['eccentricity']
# size_pep = props2_pep['area']
# nb_comp_pep = lab_pep.max()
# min_size_pep = 10
# max_size_pep = 200
# e=0.6
# mask_cv_final_pep = np.zeros(( labeled2_pep.shape ))
# for i in range(0, nb_comp_pep):
#         if (size_pep[i] >= min_size_pep and size_pep[i] <= max_size_pep and ecc_pep[i]<=e):
#             mask_cv_final_pep[labeled2_pep == i + 1] = 255



# cv2.imshow("cv pep", mask_cv_final_pep)
# cv2.imshow("pep", img_pep_rep)
# cv2.waitKey(0)          
# cv2.destroyAllWindows()




# ####
# #DISPLAYING TO CHECK ABOVE LINE IS OPTIMAL
# #cv2.imshow("blur_img4_pep", blur_img_pep_rep )
# #cv2.imshow("CV_pep", mask_cv_final_pep)
# #cv2.waitKey(0)          
# #cv2.destroyAllWindows()

# #LABELING FOR MASK 2
# labeledcv_pep= measure.label(mask_cv_final_pep)

# #FINDING PROPERTIES OF CV ONLY WITH MASK 2 
# propscv_pep = measure.regionprops_table(labeledcv_pep,img_pep_rep, 
#                                   properties=['label','area', 'mean_intensity','equivalent_diameter','bbox'])

# #INITIALIZING VALUES IN DATAFRAME FOR CV
# dfcv_pep = pd.DataFrame(propscv_pep)
# dfcv_pep['number']=dfcv_pep['label']-1

# #to decide what suitable area we should take the  
# #plt.hist(dfcv_pep['area'], bins=300, range=(0,1000))

# #FINDING INVERSE OF MASK2 AND DISPLAYING IT
# maskt_pep = mask_cv_final_pep.astype(np.uint8)
# notmask_pep= cv2.bitwise_not(maskt_pep)#check out why this is isnt working, why is it not showing the proper 
# #cv2.imshow("not_thresh_pep", notmask_pep)
# #cv2.imshow("thresh_pep",mask_cv_final_pep)
# #cv2.waitKey(0)          
# #cv2.destroyAllWindows()
# #cv2.imwrite(r"C:\Users\Nishanth\Desktop\coarcervates\test images\maskcv_pep.jpg", mask_cv_final_pep)

# #FINDING MASK 3 OUTSIDE REGION OF COARCERVATE AND DISPLAYING IT

# #
# maskt1_pep = mask_bf_final.astype(np.uint8)
# new_mask_pep= cv2.bitwise_and(maskt1_pep , notmask_pep)
# cv2.imshow("new_mask_pep", new_mask_pep)
# cv2.waitKey(0)          
# cv2.destroyAllWindows()
# #cv2.imwrite(r"C:\Users\Nishanth\Desktop\coarcervates\test images\7\ptoprmaskout_pep7.jpg", new_mask_pep)

# #LABELING THE OUTSIDE REGION
# labeled_out_pep= measure.label(new_mask_pep)

# #FINDING PROPERTIES OF THE OUTSIDE REGION
# props_out_pep= measure.regionprops_table(labeled_out_pep,img_pep_rep ,
#                                      properties=['label','area', 'mean_intensity', 'bbox'])

# #INITIALIZING THE OUTSIDE DATA FRAME, AND MAKING THE "NUMBER" COLUMN
# df_out_pep= pd.DataFrame(props_out_pep) 
# df_out_pep['number']=df_out_pep['label']-1

# dfwhole_pep['area_of_cv']=0
# dfwhole_pep['equivalent_dia_of_cv']=0
# dfwhole_pep['intensity_of_cv']=0
# dfwhole_pep['area_of_out']=0
# dfwhole_pep['intensity_of_out']=0

# # Matching the labels of the CV and the whole DROPLET 
# maxcv_pep=dfcv_pep['number'].max()
# for j in (dfwhole_pep['number']):
#     for i in (dfcv_pep['number']):
#         if (dfcv_pep['bbox-0'][i] >= dfwhole_pep['bbox-0'][j] and dfcv_pep['bbox-1'][i] >= dfwhole_pep['bbox-1'][j] and dfcv_pep['bbox-2'][i] <= dfwhole_pep['bbox-2'][j] and dfcv_pep['bbox-3'][i] <= dfwhole_pep['bbox-3'][j]):
#             dfwhole_pep['area_of_cv'][j]=dfcv_pep['area'][i]
#             dfwhole_pep['equivalent_dia_of_cv'][j]=dfcv_pep['equivalent_diameter'][i]
#             dfwhole_pep['intensity_of_cv'][j]=dfcv_pep['mean_intensity'][i]
#             break
     

# # Matching the labels of outside region of the droplet and whole DROPLET               
# maxwhole_pep=dfwhole_pep['number'].max()
# maxout_pep=df_out_pep['number'].max()
# for i in df_out_pep['number']:
#     for j in dfwhole_pep['number']:
#         if (df_out_pep['bbox-0'][i] == dfwhole_pep['bbox-0'][j] and df_out_pep['bbox-1'][i] == dfwhole_pep['bbox-1'][j] and df_out_pep['bbox-2'][i] == dfwhole_pep['bbox-2'][j] and df_out_pep['bbox-3'][i] == dfwhole_pep['bbox-3'][j]):
#             dfwhole_pep['area_of_out'][j]=df_out_pep['area'][i]          
#             dfwhole_pep['intensity_of_out'][j]=df_out_pep['mean_intensity'][i]
#             break
            
            
            

# #FINDING THE VALUES OF THE VOLUME OF THE WHOLE DROPLET AND THE VOLUME OF THE CV
# dfwhole_pep['volume']= 4/3*3.14* (dfwhole_pep['equivalent_diameter']/2)**3
# dfwhole_pep['volumeof_cv']= 4/3*3.14* (dfwhole_pep['equivalent_dia_of_cv']/2)**3

# dfwhole_pep['volume ratio pep']= dfwhole_pep['volumeof_cv']/dfwhole_pep['volume']
#######################


#CODE FOR DISPLAYING THE 3D GRAPH BTW NORMALIZED CONC OF PEPTIDE VS NORMALIZED CONC OF DNA VS VOLUME FRACTION
# Creating figure

#find avg and correct the valu of z 

'''
z=np.zeros(maxwhole+1)

for a in (dfwhole['number']):
   if (dfwhole_pep['volume ratio pep'][a] != 0 and dfwhole['volume ratio pep'][a] != 0):
       z[a]=(dfwhole_pep['volume ratio pep'][a]+dfwhole['volume ratio pep'][a])/2
   else:
       z[a]=dfwhole['volume ratio pep'][a]+dfwhole_pep['volume ratio pep'][a]
  
       
       
x = df1['Norm_conc']
y = df2['Norm_conc']

# Creating plot

ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z);

ax.set_xlabel('Norm conc of DNA', fontweight ='bold')
ax.set_ylabel('Norm conc of Peptide', fontweight ='bold')
ax.set_zlabel('Volume fraction', fontweight ='bold')
plt.show()
'''
# MAKE A DATAFRAME WITH ALL THE REQUIRED INFO

final=pd.DataFrame()

final['label']=dfwhole['label']

final['Droplet_Dia']=dfwhole['equivalent_diameter']

final['Coarcervate_Dia']=0
for a in (dfwhole['number']):
   # if (dfwhole_pep['equivalent_dia_of_cv'][a] != 0 and dfwhole['equivalent_dia_of_cv'][a] != 0):
   #     final['Coarcervate_Dia'][a]=(dfwhole_pep['equivalent_dia_of_cv'][a]+dfwhole['equivalent_dia_of_cv'][a])/2
   # else:
      final['Coarcervate_Dia'][a]=dfwhole['equivalent_dia_of_cv'][a]#+dfwhole_pep['equivalent_dia_of_cv'][a]

final['Vol_droplet']=dfwhole['volume']

final['Vol_CV']=0
for a in (dfwhole['number']):
   # if (dfwhole_pep['volumeof_cv'][a] != 0 and dfwhole['volumeof_cv'][a] != 0):
   #     final['Vol_CV'][a]=(dfwhole_pep['volumeof_cv'][a]+dfwhole['volumeof_cv'][a])/2
   # else:
      final['Vol_CV'][a]=dfwhole['volumeof_cv'][a]#+dfwhole_pep['volumeof_cv'][a]

final['Vol_out']= final['Vol_droplet']-final['Vol_CV']

final['VolRatio_CV/Whole']= final['Vol_CV']/final['Vol_droplet']
final['VolRatio_CV/Out']= final['Vol_CV'] / final['Vol_out']

final['F_bar_DNA']=df1['mean_intensity']
#final['F_bar_PLL']=df2['mean_intensity']
final['F_CV_DNA']=dfwhole['intensity_of_cv']
#final['F_CV_PLL']=dfwhole_pep['intensity_of_cv']
final['F_out_DNA']=dfwhole['intensity_of_out']
#final['F_out_PLL']=dfwhole_pep['intensity_of_out']


final['k_DNA']=float('inf')
#final['k_PLL']=float('inf')
final['k_DNA']=final['F_out_DNA'] / final['F_CV_DNA']
#final['k_PLL']=final['F_out_PLL'] / final['F_CV_PLL']

#final['k_PLL'].replace(np.nan,np.inf)


#for a in (dfwhole['number']):
#     if(final['k_PLL'][a]=="nan"):
#            final['k_PLL'][a]=1.0/0
'''
for a in (dfwhole['number']):
    if (final['F_CV_DNA'][a]!=0): 
        final['k_DNA']=final['F_out_DNA'] / final['F_CV_DNA']
    if(final['F_CV_PLL'][a]!=0):
        final['k_PLL']=final['F_out_PLL'] / final['F_CV_PLL']
'''
final['C_bar_DNA']= df1['Norm_conc']
final['C_bar_PLL']= df2['Norm_conc']

final['C_CV_DNA']=0.0
# final['C_CV_PLL']=0.0
for a in (dfwhole['number']):
    if (final['k_DNA'][a]!=float('inf')):
        final['C_CV_DNA'][a]=(final['C_bar_DNA'][a]) / (final['VolRatio_CV/Whole'][a] + final['k_DNA'][a] * (1-final['VolRatio_CV/Whole'][a]))
    # if(final['k_PLL'][a]!=float('inf')): 
    #     final['C_CV_PLL'][a]=(final['C_bar_PLL'][a]) / (final['VolRatio_CV/Whole'][a] + final['k_PLL'][a] * (1-final['VolRatio_CV/Whole'][a]))
    

final['C_out_DNA']=final['C_CV_DNA'] * final['k_DNA']
# final['C_out_PLL']=final['C_CV_PLL'] * final['k_PLL']

# for a in (dfwhole['number']):
#     if((final['C_CV_DNA'][a]==0) ^ (final['C_CV_PLL'][a]==0)):
#         final.drop(a,axis=0,inplace=True) 

#"VolRatio_CV/Whole"
#"C_bar_DNA"





final.sort_values("VolRatio_CV/Whole", axis = 0, ascending = True,
                  inplace = True, na_position ='first')

la=1
for i in final.index:
    final['label'][i]=la
    la=la+1

final.to_csv(r"C:\Users\Nishanth\Desktop\20X increasing Vol ratio\7.csv")




# final.sort_values("C_bar_DNA", axis = 0, ascending = True,
#                   inplace = True, na_position ='first')
# la=1
# for i in final.index:
#     final['label'][i]=la
#     la=la+1

# final.to_csv(r"C:\Users\Nishanth\Desktop\20X increasing Vol ratio\7_1.csv")




x1=np.float16(final['label'])
y1=np.float16(final['C_bar_DNA'])
y2=np.float16(final['C_CV_DNA'])
y3=np.float16(final['C_out_DNA'])


index = 0
idx = []
for i in range(len(y2)):
    if (y2[i] == 0.0 or math.isnan(y2[i]) or math.isnan(y3[i])):
        idx.append(index)
    index+=1
x1 = np.delete(x1, idx)
y1 = np.delete(y1, idx)
y2 = np.delete(y2, idx)
y3 = np.delete(y3, idx)

        
#plt.subplot(1, 2, 1)
plt.scatter(x1, y1,label='label vs Conc_barcode_DNA')
plt.scatter(x1, y2,label='label vs Conc_Coacervate_DNA')
plt.scatter(x1, y3,label='label vs Conc_Outside_Region_DNA')
plt.legend(loc='upper left')
plt.title('PLL') #change this back to DNA
plt.xlabel('Label Numbers')
plt.ylabel('Concentration')
plt.show()


# X1= np.float16(final['label'])  
# Y1= np.float16(final['C_bar_PLL'])  
# Y2= np.float16(final['C_CV_PLL'])  
# Y3= np.float16(final['C_out_PLL'])   

# index = 0
# idx = []
# for i in range(len(Y2)):
#     if Y2[i] == 0.0:
#         idx.append(index)
#     index+=1
# X1 = np.delete(X1, idx)
# Y1 = np.delete(Y1, idx)
# Y2 = np.delete(Y2, idx)
# Y3 = np.delete(Y3, idx)

# plt.subplot(1, 2, 2)
# plt.scatter(X1, Y1,label='label vs Conc_barcode_PLL')
# plt.scatter(X1, Y2,label='label vs Conc_Coacervate_PLL')
# plt.scatter(X1, Y3,label='label vs Conc_Outside_Region_PLL')
# plt.legend(loc='upper left')
# plt.title('PLL')
# plt.xlabel('Label Numbers')
# plt.ylabel('Concentration')
#plt.show()


