# -*- coding: utf-8 -*-
from lzma import FORMAT_ALONE
import requests

comandline0 = """SELECT TOP 10000 "II/246/out".RAJ2000, "II/246/out".DEJ2000, "II/246/out"."2MASS" FROM "II/246/out" """

name = 'ICRS'
comandline1 = f"""SELECT TOP 100 a.RAJ2000, a.DEJ2000, b.RAJ2000 , b.DEJ2000
FROM "II/246/out" AS a, "II/293/glimpse" AS b
WHERE 1=CONTAINS(POINT('{name}',b.RAJ2000,b.DEJ2000),CIRCLE('{name}',a.RAJ2000,a.DEJ2000,1/3600.))
AND 0=DISTANCE(POINT('{name}',b.RAJ2000,b.DEJ2000),POINT('{name}',a.RAJ2000,a.DEJ2000))-MIN(DISTANCE(POINT('{name}',b.RAJ2000,b.DEJ2000),POINT('{name}',a.RAJ2000,a.DEJ2000)))"""

test_commandline ="""SELECT a.RAJ2000, a.DEJ2000, b.RAJ2000, b.DEJ2000
FROM "II/246/out" AS a, "II/293/glimpse" AS b
WHERE 1=CONTAINS(POINT('ICRS',a.RAJ2000,a.DEJ2000), BOX('GALACTIC', 0, 0, 100/3600., 100/3600.)) 
AND 1=CONTAINS(POINT('ICRS',a.RAJ2000,a.DEJ2000), CIRCLE('ICRS',b.RAJ2000,b.DEJ2000, 2/3600.))
"""

test_commandline_1 ="""SELECT a.RAJ2000, a.DEJ2000
FROM "II/246/out" AS a
WHERE 1=CONTAINS(POINT('ICRS',a.RAJ2000,a.DEJ2000), BOX('ICRS', 0, 0, 1., 1.)) 
"""



import sys
sys.stderr.write(comandline1)

r = requests.post(
            'https://tapvizier.u-strasbg.fr/TAPVizieR/tap/sync',
            data={'request': 'doQuery', 'lang': 2, 'query': f'{test_commandline_1}'},
            )
#print(r.text)
f = open('temp.vot','w')
f.write(r.text)
f.close()

from astropy.io.votable import parse_single_table
table = parse_single_table('temp.vot').to_table()
table.write('temp.csv', overwrite=True)




