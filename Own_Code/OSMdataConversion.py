import geopandas as gpd
import numpy as np
import pandas as pd
import osmnx as ox

EXTENT_PATH = "C://Users//alek-//Documents//Wageningen Thesis//Study_Area_data//StudyAreaExtent.shp"

def OSMDataGrab(extent, items=True):
    if items:
        tags = {'building': True, 'highway': True, 'leisure':True, 'natural':True, 'public_transport':True, 
                'railway':True, 'waterway':True}
    else:
        tags = {'barrier': True, 'boundary': True, 'geological':True, 'landuse':True, 'man_made':True}

    osmData = ox.features.features_from_bbox(extent, tags)    
    return osmData 

class spatailLandReclassification(object):
    def init(self, path):
        self.extentPath = path
        self.extentSHP = gpd.read_file(path)
        self.OSMData  = OSMDataGrab(self.extentSHP)
        self.OSMLandInfo = OSMDataGrab(self.extentSHP, False)
        
    def evaluateLandType(self):
        ses