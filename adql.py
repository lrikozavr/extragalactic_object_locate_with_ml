# -*- coding: utf-8 -*-
from lzma import FORMAT_ALONE
import requests

comandline0 = """SELECT TOP 10000 "II/246/out".RAJ2000, "II/246/out".DEJ2000, "II/246/out"."2MASS" FROM "II/246/out" """

name = 'ICRS'
comandline1 = f"""SELECT TOP 100 a.RAJ2000, a.DEJ2000, b.RAJ2000 , b.DEJ2000
FROM "II/246/out" AS a, "II/293/glimpse" AS b
WHERE 1=CONTAINS(POINT('{name}',b.RAJ2000,b.DEJ2000),CIRCLE('{name}',a.RAJ2000,a.DEJ2000,1/3600.))
AND 0=DISTANCE(POINT('{name}',b.RAJ2000,b.DEJ2000),POINT('{name}',a.RAJ2000,a.DEJ2000))-MIN(DISTANCE(POINT('{name}',b.RAJ2000,b.DEJ2000),POINT('{name}',a.RAJ2000,a.DEJ2000)))"""

import sys
sys.stderr.write(comandline1)

r = requests.post(
            'https://tapvizier.u-strasbg.fr/TAPVizieR/tap/sync',
            data={'request': 'doQuery', 'lang': 2, 'query': f'{comandline1}'},
            )
print(r.text)

