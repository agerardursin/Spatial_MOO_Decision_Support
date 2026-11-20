# -*- coding: utf-8 -*-
"""
/***************************************************************************
 LandCoverReclassifier
                                 A QGIS plugin
 Reclassifies a raster to a UMEP land cover grid
                              -------------------
        begin                : 2015-07-13
        git sha              : $Format:%H$
        copyright            : (C) 2015 by Fredrik Lindberg
        email                : fredrikl@gvc.gu.se
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from __future__ import absolute_import
from builtins import str
from builtins import range
from builtins import object
# I'm gonna do my own thing now cause fuck it
import os
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import rioxarray as rxr
import rasterio as rio
from rasterio.plot import plotting_extent
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

IMP_PATH = "C://Users//alek-//Documents//Wageningen Thesis//Study_Area_data//Downloaded_data//Imperviousness NLCD//"
NLCD_PATH = "C://Users//alek-//Documents//Wageningen Thesis//Study_Area_data//Downloaded_data//Annual NLCD 1999-2023//"
ECO_DATA_PATH = "C://Users//alek-//Documents//Wageningen Thesis//Study_Area_data//Downloaded_data//Ecological Potential MRLC data//" #Important data however does not impact urban LA at all
TREE_CANOPY_PATH = "C://Users//alek-//Documents//Wageningen Thesis//Study_Area_data//Downloaded_data//nlcd_tcc_CONUS_all//"
LCZ_PATH = "C://Users//alek-//Documents//Wageningen Thesis//Study_Area_data//Downloaded_data//LCZ_LA_data_Saeid_Sepasizangabadi//"
OSM_DATA_PATH = "C://Users//alek-//Documents//Wageningen Thesis//Study_Area_data//Downloaded_data//OSM_data//"

IMP_TIF_PREFIX = "Annual_NLCD_ImpDsc_"
IMP_TIF_SUFFIX = "_CU_C1V0_Clipped_LA.tif"
imp_years = np.arange(1999,2024)

NLCD_TIF_PREFIX = "Annual_NLCD_LndCov_"
NLCD_TIF_SUFFIX = "_CU_C1V0_Clipped_LA.tif"
nlcd_years = np.arange(1999,2024)

TREE_CANOPY_TIF_PREFIX = "nlcd_tcc_conus_"
TREE_CANOPY_SUFFIX = "_v2021-4_Clipped_LA.tif"
tree_canopy_years = np.arange(2011,2022)

LCZ_FILE_NAME = "lcz_LA_sepasizangabadi_singleband_reproj.tif"

OSM_SHP_FILE_PREFIX = "OSM_"
OSM_SHP_FILE_SUFFIX = "_LA.shp"
OSM_SHP_PNTS_FILE = "_pnts"
OSM_SHP_LINES_FILE = "_lines"
OSM_SHP_MULTILINES_FILE = "_multilines"
OSM_SHP_POLYGON_FILE_SUFFIX = "_polygon"
OSM_SHP_FILE_TYPES = [OSM_SHP_PNTS_FILE, OSM_SHP_LINES_FILE, OSM_SHP_MULTILINES_FILE, OSM_SHP_POLYGON_FILE_SUFFIX, ""]
OSM_DATA = ["building", "crafts", "facilities", "geography", "highway_barrier","highway_passageways", "highway_streets", "highway_waypoints", "highway_ways", 
            "historic_elements", "indoor", "land_use_and_natural_elements", "man_made_elements", "money", "offices", "relations", "road", "shops", 
            "sports_leisure", "transport", "waterways"]

#TODO: Add a CRS checking function - want all the CRS to match 

class landUseRaster(object):
    def init(self, dataType, year):
        match dataType.lower():
            case "imp":
                if year < np.min(imp_years):
                    year = np.min(imp_years)
                    print("Year is earlier than we have data for. \n")

                elif year > np.max(imp_years):
                    year = np.max(imp_years)
                    print("Year is later than we have data for. \n")

                try:
                    self.path = IMP_PATH + IMP_TIF_PREFIX + str(year) + IMP_TIF_SUFFIX
                    self.year = year
                    self.raster = rxr.open_rasterio(self.path)
                except:
                    print("Could not open impervious file")
            case "nlcd":
                if year < np.min(nlcd_years):
                    year = np.min(nlcd_years)
                    print("Year is earlier than we have data for. \n")

                elif year > np.max(nlcd_years):
                    year = np.max(nlcd_years)
                    print("Year is later than we have data for. \n")

                try:
                    self.path = NLCD_PATH + NLCD_TIF_PREFIX + str(year) + NLCD_TIF_SUFFIX
                    self.year = year
                    self.raster = rxr.open_rasterio(self.path)
                except:
                    print("Could not open NLCD file")
            case "lcz":
                try:
                    self.path = LCZ_PATH + LCZ_FILE_NAME
                    self.year = 2022
                    self.raster = rxr.open_rasterio(self.path)
                except:
                    print("Could not open LCZ file")
            case "tree":
                if year < np.min(tree_canopy_years):
                    year = np.min(tree_canopy_years)
                    print("Year is earlier than we have data for. \n")

                elif year > np.max(tree_canopy_years):
                    year = np.max(tree_canopy_years)
                    print("Year is later than we have data for. \n")

                try:
                    self.path = TREE_CANOPY_PATH + TREE_CANOPY_TIF_PREFIX + str(year) + TREE_CANOPY_SUFFIX
                    self.year = year
                    self.raster = rxr.open_rasterio(self.path)
                except:
                    print("Could not open tree canopy file")