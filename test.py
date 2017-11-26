#!/usr/bin/env python

try:
    import urllib2 #python2
except:
    import urllib.request as urllib2 #python3
import sys

req = urllib2.Request(sys.argv[1], headers={'User-Agent':'Mozilla/5.0'})
urllib2.urlopen(req)