#!/bin/bash
#arg1 = "class_name"; for example "qso"
cd /media/kiril/j_08/ML/extragal/predict/
ls=$(ls)
#for _name in et_b et_not_b rf_b rf_not_b
for _name in normal dark
do
echo "RA,DEC,eRA,eDEC,plx,eplx,pmra,pmdec,epmra,epmdec,ruwe,g,bp,rp,RAw,DECw,w1,ew1,snrw1,w2,ew2,snrw2,w3,ew3,snrw3,w4,ew4,snrw4,dra,ddec,AGN_probability" > Gaia_AllWISE_50_$1_$_name.csv
done

for name_ in 00_ 1_ 2_ 3_ 4_ 5_ 6_ 7_ 8_ 9_ 10_ 11_ 12_
#for name in 10
do
#for _name in _et_b _et_not_b _rf_b _rf_not_b
for _name in _normal _dark
do
awk -F, '{file="Gaia_AllWISE_50_'$1''$_name'.csv"; if($31>0.5 && FNR!=1){print $0 >> file}}' Gaia_AllWISE_file_$name_$1$_name.csv
echo "$name done!"
wc -l Gaia_AllWISE_50_$1$_name.csv
done
done