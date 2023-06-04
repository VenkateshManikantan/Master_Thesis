import pandas as pd 
from protlearn.features import aac
from protlearn.features import aaindex1
from protlearn.features import ngram
from protlearn.features import entropy
from protlearn.features import atc
from protlearn.features import ctd
from protlearn.features import ctdc
from protlearn.features import ctdt
from protlearn.features import apaac
from protlearn.features import moreau_broto
from protlearn.features import moran
from protlearn.features import geary
from protlearn.features import paac
from protlearn.features import socn
from protlearn.features import cksaap
from protlearn.features import ctdd
from protlearn.features import qso
import os

def entropy_input(seq):
   try:
      value = entropy(seq)
   except:
      value = 'NaN'
   return value

def feature_extractions(seq):
    
    feature_extraction_0 = pd.DataFrame(columns = ['seqs'])
    feature_extraction_0['seqs'] = seq
    
    comp, aa = aac(seq, remove_zero_cols=False)
    feature_extraction = pd.DataFrame(comp ,columns = ["A","C","D","E","F","G","H","I","K","L","M",
                                                       "N","P","Q","R","S","T","V","W","Y"])
    aaind, inds = aaindex1(seq, standardize='zscore')
    feature_extraction_2 = pd.DataFrame(aaind ,columns = inds)
    
    di, ngrams = ngram(seq, n=2, method='absolute')
    feature_extraction_3 = pd.DataFrame(di ,columns = ngrams)

    tri, ngrams = ngram(seq, n=3, method='relative')
    feature_extraction_4 = pd.DataFrame(tri ,columns = ngrams)

    frames = [feature_extraction_0,feature_extraction, feature_extraction_2,feature_extraction_3,feature_extraction_4]
    extracted_data_0 = pd.concat(frames,axis =1)
    extracted_data_0["entropy"] = extracted_data_0["seqs"].apply(entropy_input)

    atoms, bonds = atc(seq)
    feature_extraction_5 = pd.DataFrame(atoms,columns = ["atom_1","atom_2","atom_3","atom_4","atom_5"])
    feature_extraction_6 = pd.DataFrame(bonds,columns = ["bond_1","bond_2","bond_3"])

    c, desc = ctdc(seq)
    feature_extraction_7 = pd.DataFrame(c,columns = desc)

    t, desc = ctdt(seq)
    feature_extraction_8 = pd.DataFrame(t,columns = desc)

    frames = [extracted_data_0,feature_extraction_5,feature_extraction_6,feature_extraction_7,feature_extraction_8]
    extracted_data_1 = pd.concat(frames,axis =1)

    mb = moreau_broto(seq) # ERROR NOTICE
    feature_extraction_9 = pd.DataFrame(mb,columns = ["M_br_1","M_br_2","M_br_3","M_br_4","M_br_5",
                                                    "M_br_6","M_br_7","M_br_8"])
    
    mo = moran(seq) 
    feature_extraction_10 = pd.DataFrame(mo,columns = ["M_o_1","M_o_2","M_o_3","M_o_4","M_o_5",
                                                    "M_o_6","M_o_7","M_o_8"])
    
    gr = geary(seq) 
    feature_extraction_11 = pd.DataFrame(gr,columns = ["g_r_1","g_r_2","g_r_3","g_r_4","g_r_5",
                                                    "g_r_6","g_r_7","g_r_8"])
    
    paac_comp, desc = paac(seq, lambda_=30, remove_zero_cols= False)
    feature_extraction_12 = pd.DataFrame(paac_comp,columns = desc)

    apaac_comp, desc = apaac(seq, lambda_=30, remove_zero_cols= False)
    feature_extraction_13 = pd.DataFrame(apaac_comp,columns = desc)

    frames = [extracted_data_1,feature_extraction_9,feature_extraction_10,feature_extraction_11,feature_extraction_12,feature_extraction_13]
    extracted_data_2 = pd.concat(frames,axis =1)

    ck, pairs = cksaap(seq, remove_zero_cols=False)
    feature_extraction_14 = pd.DataFrame(ck,columns = pairs)

    sw, g = socn(seq)

    feature_extraction_15 = pd.DataFrame(g,columns = ["g_1","g_2","g_3","g_4","g_5","g_6",
                                                     "g_7","g_8","g_9","g_10","g_11","g_12",
                                                     "g_13","g_14","g_15","g_16","g_17","g_18",
                                                     "g_19","g_20","g_21","g_22","g_23","g_24",
                                                     "g_25","g_26","g_27","g_28","g_29","g_30"])
    
    feature_extraction_16 = pd.DataFrame(sw,columns =["sw_1","sw_2","sw_3","sw_4","sw_5","sw_6",
                                                     "sw_7","sw_8","sw_9","sw_10","sw_11","sw_12",
                                                     "sw_13","sw_14","sw_15","sw_16","sw_17","sw_18",
                                                     "sw_19","sw_20","sw_21","sw_22","sw_23","sw_24",
                                                     "sw_25","sw_26","sw_27","sw_28","sw_29","sw_30"])

    d, desc = ctdd(seq)
    feature_extraction_17 = pd.DataFrame(d,columns = desc)

    qsw, qg, desc = qso(seq, d=30, remove_zero_cols=False)
    feature_extraction_18 = pd.DataFrame(qsw,columns = desc)
    feature_extraction_19 = pd.DataFrame(qg,columns = desc)

    frames = [extracted_data_2,feature_extraction_14,feature_extraction_15,feature_extraction_16,feature_extraction_17,feature_extraction_18,feature_extraction_19]
    extracted_data_3 = pd.concat(frames,axis =1)

    extracted_data_3.to_csv("data/run_ext_data_0.csv")
    extracted_data_3 = pd.read_csv("data/run_ext_data_0.csv")
    my_file = "data/run_ext_data_0.csv"
    if os.path.exists(my_file):
               os.remove(my_file)
    return extracted_data_3