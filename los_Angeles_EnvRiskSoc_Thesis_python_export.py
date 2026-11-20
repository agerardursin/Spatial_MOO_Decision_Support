# %% [markdown]
# # Detection of Spatial Signatures (define) in Los Angeles
# 
# This notebook is derived from the original notebook from Darribas' work that detected the spatial signatures of Houston, limited by its functional urban area. It was modified for Los Angeles and combined with code to compute social sustainabiltiy indicators based on concepts of interconnectivity, and to include results from an environmental prediction model. 
# 
# Spatial Signatures
# Environmental Risk
# 
# 
# This notebook, similar to the notebooks created as part of the Urban grammar project, relies on a version of the `gds_env` as its main computational engine, which is deployed on a variety of servers and setups. However, due to some outdated elements in the original code, and the need for more libraries to be added to the container, the current version is called 'gds_env:mod'. However, using this container,all the code produced in the project is reproducible, however when initializing the container, one must bind it to their own local work directory, as well as change all the references to folders to ones representative of their own machine. This notebook uses `gds_py` flavour in a version `6.0` as the base that was futher modified to run with the updated code.
# 
# The docker image used for this work can be found in the `\software_exploration\` directory, called `gds_mod_AGU.tar` and can be recreated using the `docker load < gds_mod_AGU.tar` command.
# 
# For more information on the generic `gds_env`, please visit:
# 
# > https://darribas.org/gds_env/
# 
# For information on `docker load`, please visit:
# 
# > https://docs.docker.com/reference/cli/docker/image/load/
# 
# - 1. Analysis of form
#     - 1.1. Data retrieval
#         - 1.1.1. Bulding footprints
#         - 1.1.2. Street network
#         - 1.1.3. Spatial barriers
#     - 1.2. Data pre-processing
#         - 1.2.1. Footprints checks and cleaning
#         - 1.2.2. Barries checks and cleaning
#     - 1.3. Generation of geographies
#         - 1.3.1. Enclosures
#         - 1.3.2. Enlclosed tessellation
#     - 1.4. Morphometric analysis
#         - 1.4.1. Primary morphometric characters
#         - 1.4.2. Contextualisation
# - 2. Analysis of function
#     - 2.1. Population
#     - 2.2. Land cover
#     - 2.3. Night lights
#     - 2.4. Workplace characteristics
#     - 2.5. NDVI
#     - 2.6. Historical sites
# - 3. Cluster analysis
#     - 3.1. Form
#     - 3.2. Function
#     - 3.3. Spatial Signatures
# - 4. Figures
# - 5. Cluster summary
# 
# All packages and custom modules are imported below

# %%
import sys
sys.path.append("work/spatial_signatures_concept/code/")
import glob
import pickle
import warnings
from time import time

import dask.dataframe as dd
import dask_geopandas
import fiona
import geopandas as gpd
import libpysal
import momepy
import networkx as nx
import numpy as np
import osmnx as ox
import pandana
import pandas as pd
import pygeos
import pyrosm
import rasterstats
import rioxarray
import scipy
import shapely
import utils
import utils.consolidate
import utils.momepy
import utils.tobler
import xarray
from clustergram import Clustergram
from dask.distributed import Client, LocalCluster, as_completed
from download import download
from rioxarray.merge import merge_arrays
from shapely.geometry import box, mapping
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from tqdm import tqdm
from utils.momepy_utils import (_circle_radius, centroid_corner, elongation,
                                get_corners, get_edge_ratios, get_nodes,
                                solar_orientation_poly, squareness)
from utils.dask_geopandas import dask_dissolve

import urbangrammar_graphics as ugg
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from tobler.area_weighted import area_interpolate
import contextily

from esda.moran import Moran, Moran_Local_BV, Moran_Local, Moran_BV
from splot import esda as esdaplot

try:
  import yellowbrick
except:
  !pip install yellowbrick
  import yellowbrick

print("Done")

# %%
warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

# %% [markdown]
# First we specify a path to a folder for a data storage.

# %%
folder = "work/data_for_spatial_sig/output/" 
SA_folder = "work/data_for_spatial_sig/UMEP/SA_outputs/"
SA_IN_folder = "work/data_for_spatial_sig/UMEP/SA_Inputs/"
UMEP_ANALYSIS_folder ="work/data_for_spatial_sig/UMEP/"
SOC_ANALYSIS_folder ="work/data_for_spatial_sig/socialSustainability/"

# %%
study_area = gpd.read_file("work/data_for_spatial_sig/LosAngelesExtent/StudyAreaExtent_v2.shp",  enabled_drivers=["ESRI Shapefile"])  
study_area_4326 = study_area.to_crs(4326)
study_area_3857 = study_area.to_crs(3857)


# %%
study_area_4326.geometry = study_area_4326.buffer(0)  # fix geometry topology 
polygon = study_area_4326.geometry.iloc[0] 

# %% [markdown]
# ## 1. Analysis of form
# 
# First step towards spatial signtures is an analysis of urban form to capture its patterns.
# 

# %% [markdown]
# 
# ### 1.1. Data retrieval
# 
# All data used in the work are obtained from open access sources.
# 

# %% [markdown]
# 
# #### 1.1.1 Building footprints
# 
# The building footprints will be based on OSM data and derived from OSMnx library. This was done in the other jupyter notebook for this project.

# %%
BUILDING_FOLDER_PATH = "work/data_for_spatial_sig/BuildingFootprint/"
BUILDING_FOOTPRINT_PATH = BUILDING_FOLDER_PATH + "OSMBuildingsPolyV3.shp"

# %%
buildings = gpd.read_file(BUILDING_FOOTPRINT_PATH,  enabled_drivers=["ESRI Shapefile"]) # this stage to read building footprint -> polygon is not necessary

# %% [markdown]
# We check geometry validity and fix it.

# %%
buildings.is_valid.all()

# %%
buildings.geometry

# %%
buildCopy = buildings.copy()
buildCopy.geometry = buildings.buffer(0)

# %%
buildCopy.geometry

# %%
buildCopy.is_valid.all()

# %% [markdown]
# Finally, we assign unique ID to each row.

# %%
buildCopy['uID'] = range(len(buildCopy))

# %%
buildCopy.to_parquet(folder + "buildings.pq")

# %% [markdown]
# #### 1.1.2 Street network (Done in similar way as original script)
# 
# Street network is obtined from OSM using OSMnx, and tags from the `highway` key. More detail can be found here: 
# https://wiki.openstreetmap.org/wiki/Key:highway
# 
# We get a graph, project it, turn into undirected one and export to GeoDataFrame. Then we make sure `highway` parameter is string and can be saved to file.

# %%
type_filter = '["highway"~"living_street|motorway|motorway_link|pedestrian|primary|primary_link|residential|secondary|secondary_link|service|tertiary|tertiary_link|trunk|trunk_link"]'

streets_graph = ox.graph_from_polygon(polygon, network_type='all_private', custom_filter=type_filter, retain_all=True)
streets_graph = ox.projection.project_graph(streets_graph)
gdf = ox.graph_to_gdfs(ox.get_undirected(streets_graph), nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
gdf.highway = gdf.highway.astype(str)
path = folder + 'roads_osm.pq'
gdf[['highway', 'geometry']].to_parquet(path)

# %% [markdown]
# #### 1.1.3. Spatial barriers (Done in similar way as original script)
# 
# Identification of enclosures should optimally contain variety of barriers, not only street networks. Therefore, we download railway and water geometry from OSM.
# 
# ##### Water

# %%
tags = {'natural': ['water', 'coastline', 'spring']}
gdf = ox.geometries_from_polygon(polygon, tags)
path = folder + 'water_osm.pq'
gdf[['natural', 'geometry']].to_parquet(path)

# %% [markdown]
# ##### Railway
# 
# Then we get railway and filter out proper geometry type (we want LineStrings representing railway tracks) and remove tunnels (those are not spatial barriers)..

# %%
tags = {'railway': True}
gdf = ox.geometries_from_polygon(polygon, tags)
gdf = gdf[gdf.geom_type == 'LineString']
gdf = gdf[gdf.tunnel != 'yes']
path = folder + 'railway_osm.pq'
gdf[['railway', 'geometry']].to_parquet(path)

# %% [markdown]
# ### 1.2. Data pre-processing
# 
# Before we can analyse form using the data we just downloaded, we need to ensure that they are in a proper form.
# 
# #### 1.2.1. Footprints checks and cleaning (Done in similar way as original script)
# 
# Let's start with building footprints.

# %%
buildCopy.geom_type.value_counts()

# %% [markdown]
# We explode geometries and keep only polygons.

# %%
buildCopy = buildCopy.explode(index_parts=True).reset_index(drop=True)

# %% [markdown]
# We reproject geometries to our selected CRS.

# %%
buildCopy = buildCopy.to_crs(3857)

# %%
buildCopy.is_valid.all()

# %% [markdown]
# Since we have exploded some MultiPolygons, let's create fresh ID.

# %%
buildCopy['uID'] = range(len(buildCopy))

# %%
buildCopy.to_parquet(folder + "buildings.pq")

# %% [markdown]
# #### 1.2.2. Barries checks and cleaning (Done in similar way as original script)
# 
# Let's now check our barriers and close potential unwanted gaps to generate clean enclosures

# %%
railway = gpd.read_parquet(folder + 'railway_osm.pq').to_crs(buildCopy.crs)

# %%
railway.geom_type.unique()

# %%
roads = gpd.read_parquet(folder + 'roads_osm.pq').to_crs(buildCopy.crs)

# %% [markdown]
# Roads seems to be fine, similarly to railway. The only step we do is extension of railway lines to snap to roads.

# %%
%time extended_railway = utils.momepy.extend_lines(railway, 30, target=roads, extension=.1)

# %%
water = gpd.read_parquet(folder + "water_osm.pq").to_crs(buildCopy.crs)

# %%
water.geom_type.value_counts()

# %% [markdown]
# We do not want point water features and for polygons, we want their boudaries.

# %%
water = water[~(water.geom_type == "Point")].explode(index_parts=True)
water.loc[water.geom_type == "Polygon", "geometry"] = water.loc[water.geom_type == "Polygon", "geometry"].boundary

# %% [markdown]
# ### 1.3. Generation of geographies (Done in similar way as original script)
# 
# Now we have to create geographies we will be using in the analysis. Enclosures and enclosed tessellation.
# 
# #### 1.3.1. Enclosures
# 
# Enclosures require spatial barriers, which are roads, water and railway (we use the extended one we did above), limited by study area.

# %%
limit = study_area_4326.to_crs(buildCopy.crs).geometry.iloc[[0]]

# %%
%%time
enclosures = utils.momepy.enclosures(roads, limit=limit, additional_barriers=[extended_railway, water])

# %%
enclosures.to_parquet(folder + "enclosures.pq")

# %% [markdown]
# #### 1.3.2. Enclosed tessellation
# 
# With enclosures and buildings, we can now create enclosed tessellation. To have a better control of dask cluster, we can create it directly (enclosed tessellation is a dask-parallelised function).

# %%
enclosures = gpd.read_parquet(folder + "enclosures.pq")
buildCopy = gpd.read_parquet(folder + "buildings.pq")

# %%
%%time
tess = utils.momepy.Tessellation(buildCopy, 'uID', enclosures=enclosures, n_chunks=16)

# %% [markdown]
# Now we can assign unique ID to each tessellation cell and save them. 

# %%
# warnings.filterwarnings(action='once')
tessellation = tess.tessellation
tessellation['tID'] = range(len(tessellation))
tessellation.to_parquet(folder + "tessellation.pq")

# %% [markdown]
# ### 1.4. Morphometric analysis
# 
# Morpohmetric analysis first measures primary characters, which are then contextualised to capture the tendencies within the local context of each tessellation cell.
# 
# #### 1.4.1. Primary morphometric characters (Slightly modified from original script due to issues with contigencies)
# 
# For the analysis we need four input layer representing urban form - buildings, streets, enclosed tessellation and enclosures.

# %%
tess = gpd.read_parquet(folder + "tessellation.pq")
enclosures = gpd.read_parquet(folder + "enclosures.pq")
buildings = gpd.read_parquet(folder + "buildings.pq")

# %%
tess2 = tess.copy()

# %% [markdown]
# In many cases, we will need to relate buildings to tessellation cells. the best way to ensure the proper linkage is to combine both GeoDataFrames into a single one with two named geometry columns. Then we can assign unique ID to each tessellation cell. Remember that not every cell has assigned a building.

# %%
tess = tess.rename_geometry("tessellation").merge(
    buildings[["uID", "geometry"]].rename_geometry("buildings"), on="uID", how="left"
)

# %% [markdown]
# Dask is to paralellise as much as we can and is a holdover from the original research. However this is only done for the initial calculations before it was determined to be unnecessary for the amount of data points we have.

# %% [markdown]
# We create dask.GeoDataFrame from our geodataframe with number of partitions equal to number of workers in a cluster. 
# 
# This was found later to not be necessary, but given how much of the code had been originally built for dask, maintaining it was considered simpler even if it was somewhat overkill.

# %%
ddf = dask_geopandas.from_geopandas(tess, npartitions=16)

# %% [markdown]
# You may see in other notebooks in this project, that the following section is using `momepy` to measure morphometric characters. The following cell contains the same functions, just written for dask (momepy works with pure GeoPandas). Refer to the [documentation of momepy](https://docs.momepy.org) for details on each one of them. Each result is assigned as a new column.

# %%
%%time
## Measure morphometric characters
# Building area
ddf['sdbAre'] = ddf.buildings.area

# Building perimeter
ddf['sdbPer'] = ddf.buildings.length

# Courtyard area
exterior_area = ddf.buildings.map_partitions(lambda series: shapely.area(shapely.polygons(series.exterior.to_numpy())))#, meta='float')
ddf['sdbCoA'] = exterior_area - ddf['sdbAre']

# %%
%%time
# Circular compactness
hull = ddf.buildings.convex_hull.exterior
hull

# %%
radius = hull.apply(lambda g: _circle_radius(list(g.coords)) if g is not None else None, meta=(None, 'float64'))

# %%
ddf['ssbCCo'] = ddf['sdbAre'] / (np.pi * radius ** 2)

# %%
%%time
# Corners
ddf['ssbCor'] = ddf.buildings.apply(lambda g: get_corners(g), meta=(None, 'float64'))

# Squareness
ddf['ssbSqu'] = ddf.buildings.apply(lambda g: squareness(g), meta=(None, 'float64'))

# %%
%%time
# Equivalent rectangular index
bbox = ddf.buildings.apply(lambda g: g.minimum_rotated_rectangle if g is not None else None, meta=gpd.GeoSeries())
ddf['ssbERI'] = (ddf['sdbAre'] / bbox.area).pow(1./2) * (bbox.length / ddf['sdbPer'])

# Elongation
ddf['ssbElo'] = bbox.map_partitions(lambda s: elongation(s), meta=(None, 'float64'))

# %%
%%time
# Centroid corner mean distance and deviation
def _centroid_corner(series):
    ccd = series.apply(lambda g: centroid_corner(g))
    return pd.DataFrame(ccd.to_list(), index=series.index)


ddf[['ssbCCM', 'ssbCCD']] = ddf.buildings.map_partitions(_centroid_corner, meta=pd.DataFrame({0: [0.1], 1: [1.1]}))

# %%
%%time
# Solar orientation
ddf['stbOri'] = bbox.apply(lambda g: solar_orientation_poly(g), meta=(None, 'float64'))

# Tessellation longest axis length
hull = ddf.tessellation.convex_hull.exterior

# %%
hull

# %%
%%time
ddf['sdcLAL'] = hull.apply(lambda g: _circle_radius(list(g.coords)), meta=(None, 'float64')) * 2

# Tessellation area
ddf['sdcAre'] = ddf.tessellation.area

# %%
# Circular compactness
radius = hull.apply(lambda g: _circle_radius(list(g.coords)), meta=(None, 'float64'))
ddf['sscCCo'] = ddf['sdcAre'] / (np.pi * radius ** 2)

# Equivalent rectangular index
bbox = ddf.tessellation.apply(lambda g: g.minimum_rotated_rectangle, meta=gpd.GeoSeries())
ddf['sscERI'] = (ddf['sdcAre'] / bbox.area).pow(1./2) * (bbox.length / ddf.tessellation.length)

# %%
# Solar orientation
ddf['stcOri'] = bbox.apply(lambda g: solar_orientation_poly(g), meta=(None, 'float64'))

# Covered area ratio
ddf['sicCAR'] = ddf['sdbAre'] / ddf['sdcAre']

# Building-cell alignment
ddf['stbCeA'] = (ddf['stbOri'] - ddf['stcOri']).abs()

# %%
# Compute all characters using dask
df = ddf.compute()

# %% [markdown]
# We can save our intermediate result to a parquet.

# %%
df = df.merge(
    buildings[["building", "uID"]], on="uID", how="left"
)

# %%
df.to_parquet(folder + "tess.pq")

# %% [markdown]
# Now we use the same approach to measure characters on enclosures.

# %%
%%time
# Load data
encl = dask_geopandas.from_geopandas(enclosures, npartitions=16)

# Area
encl['ldeAre'] = encl.geometry.area

# Perimeter
encl['ldePer'] = encl.geometry.length

# Circular compacntess
hull = encl.geometry.convex_hull.exterior

radius = hull.apply(lambda g: _circle_radius(list(g.coords)) if g is not None else None, meta=(None, 'float64'))
encl['lseCCo'] = encl['ldeAre'] / (np.pi * radius ** 2)

# Equivalent rectangular index
bbox = encl.geometry.apply(lambda g: g.minimum_rotated_rectangle if g is not None else None, meta=gpd.GeoSeries())
encl['lseERI'] = (encl['ldeAre'] / bbox.area).pow(1./2) * (bbox.length / encl['ldePer'])

# Compactness-weighted axis
longest_axis = hull.apply(lambda g: _circle_radius(list(g.coords)), meta=(None, 'float64')) * 2
encl['lseCWA'] = longest_axis * ((4 / np.pi) - (16 * encl['ldeAre']) / ((encl['ldePer']) ** 2))

# Solar orientation
encl['lteOri'] = bbox.apply(lambda g: solar_orientation_poly(g), meta=(None, 'float64'))

# Compute data and return gpd.GeoDataFrame
encl_df = encl.compute()

# Weighted number of neighbors
inp, res = encl_df.sindex.query(encl_df.geometry, predicate='intersects')
indices, counts = np.unique(inp, return_counts=True)
encl_df['neighbors'] = counts - 1
encl_df['lteWNB'] = encl_df['neighbors'] / encl_df['ldePer']

# Measure weighted cells within enclosure
encl_counts = tess.groupby('eID').count()
merged = encl_df[['eID', 'ldeAre']].merge(encl_counts[['tessellation']], how='left', on='eID')
encl_df['lieWCe'] = merged['tessellation'] / merged['ldeAre']

# %%
np.min(encl_df['lieWCe'])

# %%
# Save data to parquet
encl_df.to_parquet(folder + "enclosures.pq")

# %% [markdown]
# #### Chunk the data
# 
# Some functions are hard to parallelize. Therefore we split the data into chunks and run the analysis on those. This is more important with larger pieces of data and not as much with the size of data that we used for Los Angeles. However, in following the original implementation it was considered easier to keep it rather than change.

# %%
cells = gpd.read_parquet(folder + "tess.pq")

# %%
cell_noDup = cells.copy()

# %%
# AGU Addition - Shannon diversity of building types - not used in final report but was interesting data to examine
shannon_sw = momepy.sw_high(k=3, gdf=cell_noDup, ids='tID')
cells['bdShDv'] = momepy.Shannon(gdf=cells, values='building', spatial_weights=shannon_sw, 
                              unique_id='tID', categorical=True).series

# %%
cells.to_parquet(folder + "tess.pq")

# %%
cells.head(5)

# %%
bounds = cells.total_bounds

# %% [markdown]
# First, we create a grid over the extent.

# %%
size = 7
grid = gpd.points_from_xy(
    np.repeat(np.linspace(bounds[0], bounds[2], size), size), np.tile(np.linspace(bounds[1], bounds[3], size), size)
, crs=cells.crs)

# %%
print(grid.crs)

# %%
grid, _ = libpysal.cg.voronoi_frames(np.stack([np.repeat(np.linspace(bounds[0], bounds[2], size), size), np.tile(np.linspace(bounds[1], bounds[3], size), size)]).T, clip="bbox")

# %%
ax = _.plot()
grid.plot(ax=ax, edgecolor="k", alpha=.3)

# %% [markdown]
# We assing ID to each grid cell and spatially join them to tessellation.

# %%
grid['chunkid'] = range(len(grid))

# %%
print(grid.crs)

# %%
%time join = gpd.sjoin(gpd.GeoDataFrame(geometry=cells.tessellation.centroid), grid, how='left')

# %% [markdown]
# We save grid for later.

# %%
grid.to_parquet(folder + "chunks.pq")

# %% [markdown]
# We save chunked data to separate files.

# %%
for i in join.chunkid.unique():
    cells[join["chunkid"] == i].to_parquet(folder + f"chunked/chunk_{i}.pq")

# %% [markdown]
# We also need to know the neighbors of each chunk, to avoid boundary effect. We first create spatial weights.

# %%
w = libpysal.weights.fuzzy_contiguity(grid)

# %%
grid = grid.set_index("chunkid")

# %% [markdown]
# How many chunks we actaully have?

# %%
total = len(glob.glob(folder + "chunked/*"))
total

# %% [markdown]
# Now we loop over each chunk, get neighbouring chunks and IDs of close-by tessellation cells from each of the neighbouring chunks.

# %%
include = pd.Series(index=pd.MultiIndex.from_tuples([], names=['chunk', 'neighbors']), dtype=object)

for ch in tqdm(join.chunkid.unique(), total=total):
    target = grid.loc[ch, 'geometry']

    for n in w.neighbors[ch]:
        try:
            tess = gpd.read_parquet(folder + f"chunked/chunk_{int(n)}.pq")

            ind = tess.sindex.query(target, predicate='intersects')

            if ind.size > 0:
                incl = ind.tolist()
                for _ in range(12):
                    inp, res = tess.sindex.query(tess.geometry.iloc[incl], predicate='intersects')
                    incl += np.unique(res).tolist()
            else:
                incl = []
            include.loc[ch, int(n)] = np.unique(incl)
        except:
            pass

# %% [markdown]
# And we filter out empty lists.

# %%
include = include[include.apply(lambda x: len(x) > 0)]

# %%
includePD = pd.DataFrame(include, columns=['indices'])

# %% [markdown]
# Now we know that, for example, chunk 22 has 21 as a neighbour and we should load its cells with ID [0, 1, ... 13 ...] while measuring stuff on chunk 22. Let's save this information.

# %%
pd.DataFrame(include, columns=['indices']).to_parquet(folder + 'cross-chunk_indices.pq')

# %% [markdown]
# #### Generate spatial weights (W)
# 
# Subsequent steps will require understanding of the context of each tessellation cell in a form of spatial weights matrices (Queen contiguity and Queen contiguty of inclusive 3rd order). We generate them beforehand and store as npz files representing sparse matrix.
# 
# Each geo-chunk is loaded together with relevant cross-chunk tessellation cells (to avoid edge effect). We use dask to parallelise the iteration. Number of workers is smaller now to ensure enough memory for each chunk.

# %%
warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')
warnings.filterwarnings('ignore', message='.*Assigning CRS to a GeoDataFrame without a geometry*')

# %%
cross_chunk = pd.read_parquet(folder + 'cross-chunk_indices.pq')
chunks = gpd.read_parquet(folder + "chunks.pq")

# %% [markdown]
# First we have to specify a function doing the processing itself, where the only attribure is the chunk_id.

# %%
def generate_w(chunk_id):
    # load cells of a chunk
    try:
        cells = gpd.read_parquet(folder + f"chunked/chunk_{chunk_id}.pq")

        # add neighbouring cells from other chunks
        cross_chunk_cells = []

        for chunk, inds in cross_chunk.loc[chunk_id].indices.items():
            add_cells = gpd.read_parquet(folder + f"chunked/chunk_{chunk}.pq").iloc[inds]
            cross_chunk_cells.append(add_cells)

        df = pd.concat([cells,pd.concat(cross_chunk_cells, ignore_index=True)], ignore_index=True)

        w = libpysal.weights.Queen.from_dataframe(df, geom_col='tessellation', use_index=False)
        w3 = momepy.sw_high(k=3, weights=w)

        scipy.sparse.save_npz(folder + f"weights/w_{chunk_id}.npz", w.sparse)
        scipy.sparse.save_npz(folder + f"weights/w3_{chunk_id}.npz", w3.sparse)

        return f"Chunk {chunk_id} processed sucessfully."
    except FileNotFoundError:
        return f"Chunk {chunk_id} does not exist. Skipped."

# %% [markdown]
# Then we use dask to iterate over all chunks. The following script sends first 8 chunks to dask together and then submits a new chunk as soon as any of previous finishes. That way we process only 8 chunks at once ensuring that we the cluster will not run out of memory. 
# 
# 
# There is an issue with their dask implementation -> will not use it instead just for loop for each chunk As seen in the next cell. This could be related to the issues present in maintenance of the code in general.

# %%
%%time
inputs = iter(range(49))

for id in inputs:
    generate_w(id)


# %% [markdown]
# Not all IDs in the range are actually used as chunks. Let's find out actual chunks.

# %%
real_chunks = [int(i[-5:-3].replace("_", "")) for i in glob.glob(folder + "chunked/*")]

# %% [markdown]
# Some of them failed due to memory demands. Which?

# %%
done = [int(i[-6:-4].replace("_", "")) for i in glob.glob(folder + "weights/w3_*")]
missing = [i for i in real_chunks if i not in done]

# %% [markdown]
# We can make a new cluster with less workers to give each more memory. [Addition] Since we manually iterated through all chunks, the next bit is unnecessary and thus commented out.

# %% [markdown]
# #### Spatial distribution and network analysis
# 
# To measure spatial distribution use the implementation from their script without parallelization.

# %% [markdown]
# Within the same function below we measure spatial distribution of elements and network-based characters.

# %%
def measure(chunk_id):
    # try:
    # load cells of a chunk
    cells = gpd.read_parquet(folder + f"chunked/chunk_{chunk_id}.pq")
    cells['keep'] = True

    # add neighbouring cells from other chunks
    cross_chunk_cells = []

    for chunk, inds in cross_chunk.loc[chunk_id].indices.items():
        add_cells = gpd.read_parquet(folder + f"chunked/chunk_{chunk}.pq").iloc[inds]
        add_cells['keep'] = False
        cross_chunk_cells.append(add_cells)

    df = pd.concat([cells, pd.concat(cross_chunk_cells, ignore_index=True)], ignore_index=True)

    # read W
    w = libpysal.weights.WSP(scipy.sparse.load_npz(folder + f"weights/w_{chunk_id}.npz")).to_W()

    # alignment
    def alignment(x, orientation='stbOri'):
        orientations = df[orientation].iloc[w.neighbors[x]]
        return abs(orientations - df[orientation].iloc[x]).mean()

    df['mtbAli'] = [alignment(x) for x in range(len(df))]

    # mean neighbour distance
    def neighbor_distance(x):
        geom = df.buildings.iloc[x]
        if geom is None:
            return np.nan
        return df.buildings.iloc[w.neighbors[x]].distance(df.buildings.iloc[x]).mean()

    df['mtbNDi'] = [neighbor_distance(x) for x in range(len(df))]

    # weighted neighbours
    df['mtcWNe'] = pd.Series([w.cardinalities[x] for x in range(len(df))], index=df.index) / df.tessellation.length

    # area covered by neighbours
    def area_covered(x, area='sdcAre'):
        neighbours = [x]
        neighbours += w.neighbors[x]

        return df[area].iloc[neighbours].sum()

    df['mdcAre'] = [area_covered(x) for x in range(len(df))]

    # read W3
    w3 = libpysal.weights.WSP(scipy.sparse.load_npz(folder + f"weights/w3_{chunk_id}.npz")).to_W()

    # weighted reached enclosures
    def weighted_reached_enclosures(x, area='sdcAre', enclosure_id='eID'):
        neighbours = [x]
        neighbours += w3.neighbors[x]

        vicinity = df[[area, enclosure_id]].iloc[neighbours]

        return vicinity[enclosure_id].unique().shape[0] / vicinity[area].sum()

    df['ltcWRE'] = [weighted_reached_enclosures(x) for x in range(len(df))]

    # mean interbuilding distance
    # define adjacency list from lipysal
    adj_list = w.to_adjlist(remove_symmetric=False)
    adj_list["weight"] = (
        df.buildings.iloc[adj_list.focal]
        .reset_index(drop=True)
        .distance(df.buildings.iloc[adj_list.neighbor].reset_index(drop=True)).values
    )

    G = nx.from_pandas_edgelist(
            adj_list, source="focal", target="neighbor", edge_attr="weight"
        )
    ibd = []
    for i in range(len(df)):
        try:
            sub = nx.ego_graph(G, i, radius=3)
            ibd.append(np.nanmean([x[-1] for x in list(sub.edges.data('weight'))]))
        except:
            ibd.append(np.nan)

    df['ltbIBD'] = ibd

    # Reached neighbors and area on 3 topological steps on tessellation
    df['ltcRea'] = [w3.cardinalities[i] for i in range(len(df))]
    df['ltcAre'] = [df.sdcAre.iloc[w3.neighbors[i]].sum() for i in range(len(df))]

    # Save cells to parquet keeping only within-chunk data not the additional neighboring
    df[df['keep']].drop(columns=['keep']).to_parquet(folder + f"chunked/chunk_{chunk_id}.pq")

    # Load street network for an extended chunk area
    chunk_area = chunks.geometry.iloc[chunk_id].buffer(5000)  # we extend the area by 5km to minimise edge effect
    streets = gpd.read_parquet(folder + 'roads_osm.pq').to_crs(df.crs)
    streets = gpd.clip(streets, chunk_area).explode(index_parts=True)

    # Street profile (measures width, width deviation and openness)
    sp = momepy.StreetProfile(streets, df.set_geometry("buildings"))
    streets['sdsSPW'] = sp.w
    streets['sdsSWD'] = sp.wd
    streets['sdsSPO'] = sp.o

    # Street segment length
    streets['sdsLen'] = streets.length

    # Street segment linearity
    streets['sssLin'] = momepy.Linearity(streets).series

    # Convert geopadnas.GeoDataFrame to networkx.Graph for network analysis
    G = momepy.gdf_to_nx(streets)

    # Node degree
    G = momepy.node_degree(G)

    # Subgraph analysis (meshedness, proportion of 0, 3 and 4 way intersections, local closeness)
    G = momepy.subgraph(
        G,
        radius=5,
        meshedness=True,
        cds_length=False,
        mode="sum",
        degree="degree",
        length="mm_len",
        mean_node_degree=False,
        proportion={0: True, 3: True, 4: True},
        cyclomatic=False,
        edge_node_ratio=False,
        gamma=False,
        local_closeness=True,
        closeness_weight="mm_len",
        verbose=False
    )

    # Cul-de-sac length
    G = momepy.cds_length(G, radius=3, name="ldsCDL", verbose=False)

    # Square clustering
    G = momepy.clustering(G, name="xcnSCl")

    # Mean node distance
    G = momepy.mean_node_dist(G, name="mtdMDi", verbose=False)

    # AGU addition - closeness centrality of nodes
    G = momepy.closeness_centrality(G, name="clsCnt", verbose=False)

    # AGU addition - betweenness centrality of nodes
    G = momepy.betweenness_centrality(G, name="btwCnt", verbose=False)

    # AGU addition - straightness centrality of nodes
    G = momepy.straightness_centrality(G, name="strCnt", verbose=False)

    # Convert networkx.Graph back to GeoDataFrames and W (denoting relationships between nodes)
    nodes, edges, sw = momepy.nx_to_gdf(G, spatial_weights=True)

    # Generate inclusive higher order weights
    edges_w3 = momepy.sw_high(k=3, gdf=edges)

    # Mean segment length
    edges["ldsMSL"] = momepy.SegmentsLength(edges, spatial_weights=edges_w3, mean=True, verbose=False).series

    # Generate inclusive higher order weights
    nodes_w5 = momepy.sw_high(k=5, weights=sw)

    # Node density
    nodes["lddNDe"] = momepy.NodeDensity(nodes, edges, nodes_w5, verbose=False).series

    # Weighter node density
    nodes["linWID"] = momepy.NodeDensity(nodes, edges, nodes_w5, weighted=True, node_degree="degree", verbose=False).series

    # Save to parquets
    edges.to_parquet(folder + f"edges/edges_{chunk_id}.pq")
    nodes.to_parquet(folder + f"nodes/nodes_{chunk_id}.pq")


    return f"Chunk {chunk_id} processed sucessfully."
    # except FileNotFoundError:
    #     return f"Chunk {chunk_id} does not exist. Skipped."

# %% [markdown]
# ##### Simple fix for rerunning code

# %%
tessBuild = gpd.read_parquet(folder + "tess.pq")
tessBuild.head(1)

# %%
inputs = iter(range(49))

for id in inputs:
    cells = gpd.read_parquet(folder + f"chunked/chunk_{id}.pq")
    inp2 = iter(range(len(cells)))
    # buildList=pd.DataFrame()
    for id2 in inp2:
        tessBuildLoc=tessBuild.loc[tessBuild['tID'] == cells['tID'][id2]]
        if id2 == 0:
            buildList = pd.DataFrame(tessBuildLoc[["tID","buildings"]])
        else:
            buildList = pd.concat([buildList,tessBuildLoc[["tID","buildings"]]])
    buildList.reset_index(drop=True, inplace=True)
    buildGPD = gpd.GeoDataFrame(data=buildList["tID"],geometry=buildList["buildings"])
    buildGPD.rename(columns={"geometry":"buildings"})
    
    cells = cells.merge(buildGPD, on="tID")
    cells.to_parquet(folder + f"chunked/chunk_{id}.pq")


# %% [markdown]
# ##### normal code (again)

# %%
measure(0)

# %%
%%time
inputs = iter(range(49))

for id in inputs:
    if id == 0:
        continue
    measure(id)

# %% [markdown]
# #### Link together
# 
# For the further analysis, we need to link data measured on individual elements together. We link cells to edges based on the proportion of overlap (if a cell intersects more than one edge) and nodes based on proximity (with a restriction - node has to be on linked edge). Enclosures are linked based on enclosure ID.

# %%
def link(chunk_id):
    try:
        s = time()
        cells = gpd.read_parquet(folder + f"chunked/chunk_{chunk_id}.pq")
        edges = gpd.read_parquet(folder + f"edges/edges_{chunk_id}.pq")
        nodes = gpd.read_parquet(folder + f"nodes/nodes_{chunk_id}.pq")

        cells['edgeID'] = get_edge_ratios(cells, edges)
        cells['nodeID'] = get_nodes(cells, nodes, edges, 'nodeID', 'edgeID', 'node_start', 'node_end')

        characters = ['sdsSPW', 'sdsSWD', 'sdsSPO', 'sdsLen', 'sssLin', 'ldsMSL']
        l = []
        for d in cells.edgeID:
            l.append((edges.iloc[list(d.keys())][characters].multiply(list(d.values()), axis='rows')).sum(axis=0))
        cells[characters] = pd.DataFrame(l, index=cells.index)

        cells = cells.merge(nodes.drop(columns=['geometry']), on='nodeID', how='left')
        cells = cells.rename({'degree': 'mtdDeg', 'meshedness': 'lcdMes', 'proportion_3': 'linP3W', 'proportion_4': 'linP4W',
                         'proportion_0': 'linPDE', 'local_closeness': 'lcnClo'}, axis='columns')

        cells['edgeID_keys'] = cells.edgeID.apply(lambda d: list(d.keys()))
        cells['edgeID_values'] = cells.edgeID.apply(lambda d: list(d.values()))

        cells.drop(columns='edgeID').to_parquet(folder + f"chunked/chunk_{chunk_id}.pq")

        return f"Chunk {chunk_id} processed sucessfully in {time() - s} seconds."
    except FileNotFoundError:
        return f"Chunk {chunk_id} does not exist. Skipped."

# %%
link(0)

# %%
inputs = iter(range(49))

for id in inputs:
    if id == 0:
        continue
    link(id)

# %% [markdown]
# Again did not use dask in this case either.

# %% [markdown]
# Enclosures are linked via simple attribute join and since the operation is does not require any computation, it is done as a simple loop.Now we just link data from enclosures to tessellation.

# %%
enclosures = pd.read_parquet(folder + "enclosures.pq")
for chunk_id in real_chunks:
    s = time()
    cells = gpd.read_parquet(folder + f"chunked/chunk_{chunk_id}.pq")

    cells = cells.merge(enclosures.drop(columns=['neighbors']), on='eID', how='left')

    cells.to_parquet(folder + f"chunked/chunk_{chunk_id}.pq")

    print(f"Chunk {chunk_id} processed sucessfully in {time() - s} seconds.")

# %% [markdown]
# #### Inter-element characters
# 
# The remaining morphometric characters are based on a relations between multiple elements. The implementation mirrors the approach above.

# %%
def measure(chunk_id):
    try:
        s = time()
        # Load data
        cells = gpd.read_parquet(folder + f"chunked/chunk_{chunk_id}.pq")
        edges = gpd.read_parquet(folder + f"edges/edges_{chunk_id}.pq")
        nodes = gpd.read_parquet(folder + f"nodes/nodes_{chunk_id}.pq")

        # Street Alignment
        edges['orient'] = momepy.Orientation(edges, verbose=False).series
        edges['edgeID'] = range(len(edges))
        keys = cells.edgeID_values.apply(lambda a: np.argmax(a))
        cells['edgeID_primary'] = [inds[i] for inds, i in zip(cells.edgeID_keys, keys)]
        cells['stbSAl'] = momepy.StreetAlignment(cells, 
                                                 edges, 
                                                 'stbOri', 
                                                 left_network_id='edgeID_primary', 
                                                 right_network_id='edgeID').series

        # Area Covered by each edge
        vals = {x:[] for x in range(len(edges))}
        for i, keys in enumerate(cells.edgeID_keys):
            for k in keys:
                vals[k].append(i)
        area_sums = []
        for inds in vals.values():
            area_sums.append(cells.sdcAre.iloc[inds].sum())
        edges['sdsAre'] = area_sums

        # Building per meter
        bpm = []
        for inds, l in zip(vals.values(), edges.sdsLen):
            bpm.append(cells.buildings.iloc[inds].notna().sum() / l if len(inds) > 0 else 0)
        edges['sisBpM'] = bpm

        # Cell area
        nodes['sddAre'] = nodes.nodeID.apply(lambda nid: cells[cells.nodeID == nid].sdcAre.sum())

        # Area covered by neighboring edges + count of reached cells
        edges_W = libpysal.weights.Queen.from_dataframe(edges, use_index=False)

        areas = []
        reached_cells = []
        for i in range(len(edges)):
            neighbors = [i] + edges_W.neighbors[i]
        #     areas
            areas.append(edges.sdsAre.iloc[neighbors].sum())
        #     reached cells
            ids = []
            for n in neighbors:
                 ids += vals[n]
            reached_cells.append(len(set(ids)))

        edges['misCel'] = reached_cells
        edges['mdsAre'] = areas

        # Area covered by neighboring (3 steps) edges + count of reached cells
        edges_W3 = momepy.sw_high(k=3, weights=edges_W)

        areas = []
        reached_cells = []
        for i in range(len(edges)):
            neighbors = [i] + edges_W3.neighbors[i]
        #     areas
            areas.append(edges.sdsAre.iloc[neighbors].sum())
        #     reached cells
            ids = []
            for n in neighbors:
                 ids += vals[n]
            reached_cells.append(len(set(ids)))

        edges['lisCel'] = reached_cells
        edges['ldsAre'] = areas

        # Link together 
        e_to_link = ['sdsAre', 'sisBpM', 'misCel', 'mdsAre', 'lisCel', 'ldsAre']
        n_to_link = 'sddAre'

        if 'sddAre' not in cells.columns:
            cells = cells.merge(nodes[['nodeID', 'sddAre']], on='nodeID', how='left')

        l = []
        for keys, values in zip(cells.edgeID_keys, cells.edgeID_values):
            l.append((edges.iloc[keys][e_to_link].multiply(values, axis='rows')).sum(axis=0))  # weighted by the proportion
        cells[e_to_link] = pd.DataFrame(l, index=cells.index)

        # Reached neighbors and area on 3 topological steps on tessellation
        cells['keep'] = True

        # add neighbouring cells from other chunks
        cross_chunk_cells = []

        for chunk, inds in cross_chunk.loc[chunk_id].indices.items():
            add_cells = gpd.read_parquet(folder + f"chunked/chunk_{chunk}.pq").iloc[inds]
            add_cells['keep'] = False
            cross_chunk_cells.append(add_cells)

        df = pd.concat([cells,pd.concat(cross_chunk_cells, ignore_index=True)], ignore_index=True)

        w3 = libpysal.weights.WSP(scipy.sparse.load_npz(folder + f"weights/w3_{chunk_id}.npz")).to_W()

        # Reached cells in 3 topological steps
        df['ltcRea'] = [w3.cardinalities[i] for i in range(len(df))]

        # Reached area in 3 topological steps
        df['ltcAre'] = [df.sdcAre.iloc[w3.neighbors[i]].sum() for i in range(len(df))]

        # Save
        df[df['keep']].drop(columns=['keep']).to_parquet(folder + f"chunked/chunk_{chunk_id}.pq")

        return f"Chunk {chunk_id} processed sucessfully in {time() - s} seconds."
    except FileNotFoundError:
        return f"Chunk {chunk_id} does not exist. Skipped."

# %%
measure(0)

# %%
inputs = iter(real_chunks)

for id in inputs:
    if id == 0:
        continue
    measure(id)

# %% [markdown]
# #### 1.4.2. Contextualisation (Same approach as script just updated to remove bugs in their implementation)
# 
# Cluster analysis needs to be performant on large datasets, therefore it should not contain spatial constraints (such algorithms do not scale well). At the same time, we are interested in the tendency of characters within the area. Most primary characters are local only, with minimal or no sense of context. Therefore, we do the contextualisation of each primary character to get the first, second and third quartile of the distribution of values within inclusive 10th order of contiguity around each tessellation cell, weighted by the inverse distance between cell's centroids.
# 
# We first create spatial weights of tenth order (inclusive) using the paralellised pattern we used above.

# %%
def generate_w10(chunk_id):
    w = scipy.sparse.load_npz(folder + f"weights/w_{chunk_id}.npz")
    wk = sum(map(lambda x: w ** x, range(2, 11)))
    scipy.sparse.save_npz(folder + f"weights/w10_{chunk_id}.npz", wk)
    return f"Chunk {chunk_id} processed sucessfully."

# %%
inputs = iter(real_chunks)

for id in inputs:
    generate_w10(id)


# %% [markdown]
# Then we can loop over chunks and their GeoDataFrames and measure contextualisation of each character.

# %%
characters = [
    "sdbAre",
    "sdbPer",
    "sdbCoA",
    "ssbCCo",
    "ssbCor",
    "ssbSqu",
    "ssbERI",
    "ssbElo",
    "ssbCCM",
    "ssbCCD",
    "stbOri",
    "sdcLAL",
    "sdcAre",
    "sscCCo",
    "sscERI",
    "stcOri",
    "sicCAR",
    "stbCeA",
    "mtbAli",
    "mtbNDi",
    "mtcWNe",
    "mdcAre",
    "ltcWRE",
    "ltbIBD",
    "bdShDv",
    "ltcRea",
    "ltcAre",
    "sdsSPW",
    "sdsSWD",
    "sdsSPO",
    "sdsLen",
    "sssLin",
    "ldsMSL",
    "mtdDeg",
    "lcdMes",
    "linP3W",
    "linP4W",
    "linPDE",
    "lcnClo",
    "ldsCDL",
    "xcnSCl",
    "mtdMDi",
    "clsCnt",
    "btwCnt",
    "strCnt",
    "lddNDe",
    "linWID",
    "ldeAre",
    "ldePer",
    "lseCCo",
    "lseERI",
    "lseCWA",
    "lteOri",
    "lteWNB",
    "lieWCe",
    "stbSAl",
    "sdsAre",
    "sisBpM",
    "misCel",
    "mdsAre",
    "lisCel",
    "ldsAre",
    "sddAre",
]

# %%
def convolute(chunk_id):
    # load cells of a chunk
    cells = gpd.read_parquet(folder + f"chunked/chunk_{chunk_id}.pq", columns=characters + ["tessellation", "tID"])
    cells['keep'] = True

    # add neighbouring cells from other chunks
    cross_chunk_cells = []

    for chunk, inds in cross_chunk.loc[chunk_id].indices.items():
        add_cells = gpd.read_parquet(folder + f"chunked/chunk_{chunk}.pq", columns=characters + ["tessellation", "tID"]).iloc[inds]
        add_cells['keep'] = False
        cross_chunk_cells.append(add_cells)

    data = pd.concat([cells,pd.concat(cross_chunk_cells, ignore_index=True)], ignore_index=True)
       
    gdf = gpd.GeoDataFrame(data[characters], geometry=data.tessellation.centroid)
    W = libpysal.weights.WSP(scipy.sparse.load_npz(folder + f"weights/w10_{chunk_id}.npz")).to_W()

    convolutions = {}
    for c in characters:
        convolutions[c] = []

    # measure convolutions
    for i, geom in enumerate(gdf.geometry):
        neighbours = W.neighbors[i]
        neighbours.remove(i) if i in neighbours else None
        if len(neighbours) > 0:
            vicinity = gdf.iloc[neighbours]
            distance = vicinity.distance(geom)
            distance_decay = (1 / distance).replace(np.inf, 0)  # to filter out duplications

            for c in characters:
                values = vicinity[c].values
                sorter = np.argsort(values)
                values = values[sorter]
                nan_mask = np.isnan(values)
                if nan_mask.all():
                    convolutions[c].append(np.array([np.nan] * 3))
                else:
                    sample_weight = distance_decay.values[sorter][~nan_mask]
                    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
                    weighted_quantiles /= np.sum(sample_weight)
                    interpolate = np.interp([.25, .5, .75], weighted_quantiles, values[~nan_mask])
                    convolutions[c].append(interpolate)
        else:
            for c in characters:
                convolutions[c].append(np.array([np.nan] * 3))

    conv = pd.DataFrame(convolutions, index=data.index)
    exploded = pd.concat([pd.DataFrame(conv[c].to_list(), columns=[c + '_q1', c + '_q2',c + '_q3']) for c in characters], axis=1)
    exploded.index = data.tID

    exploded.loc[(data["keep"] == True).values].to_parquet(folder + f"convolutions/convolutions_{chunk_id}.pq")
    return f"Chunk {chunk_id} processed sucessfully."

# %%
real_chunks = [int(i[-5:-3].replace("_", "")) for i in glob.glob(folder + "chunked/*")]

# %%
inputs = iter(real_chunks)

for id in inputs:
    convolute(id)


# %% [markdown]
# ## 2. Analysis of function
# 
# Next we want to characterise built environemnt by its function.
# 
# 

# %% [markdown]
# 
# ### 2.1. Amenity Diversity (using shannon) - Did not end up in final report
# Custom addition by AGU for determining the centrality of each tesselation cell as well as how close they are to a varied amount of building types.

# %%
tess = gpd.read_parquet(folder + "tess.pq")
enclosures = gpd.read_parquet(folder + "enclosures.pq")
buildings = gpd.read_parquet(folder + "buildings.pq")

# %%
tess.head(1)

# %%
tess2 = tess.merge(
    buildings[["building", "uID"]], on="uID", how="left"
)

# %%
buildings.toilets.unique()

# %%
pd.set_option("display.max_columns", None)
buildings.head(1)

# %%
inputs = iter(range(len(tess2)))

for ip in inputs:
    if tess2.building[ip] == "yes":
        tess2.building[ip] = "building"

# %%
tess2.to_parquet(folder + "tess.pq")

# %%
tess.amenity.unique()

# %%
inputs = iter(range(len(buildings)))
buildTest = buildings.copy()
buildTest.amenity_needed = buildTest.amenity.copy()

# %% [markdown]
# For interest of access to vital utilities
# - Toilets (toilets)
# - Outdoor seating (outdoor_se)
# - Cuisine
# - takeaway (mutually exclusive with cuisine)
# - internet_a and/or internet_!
# - social_fac + social_f_1 (food banks, shelters, etc.)
# - air_condit
# - shelter_ty (shelter from the sun)
# - healthcare (healthcare, healthca_1, healthca_2)
# - religion
# - shop (only convenience and supermarket)
# - amenity (only 'courthouse', 'theatre', 'library', 'fast_food', 'restaurant','fire_station', 'pub', 'toilets', 'place_of_worship', 'social_facility', 'community_centre', 'police', 'cafe', 'clinic', 'bank', 'post_office', 'bar', 'dispatch_center', 'college', 'university', 'shelter', 'marketplace', 'townhall')
# 
# 
# Thus there is a need to assess how important or useful a building could be to a person. One way of doing that is to see if they have facilities (such as toilets, internet access, air conditioning, shelter, outdoor seating) or if they are vital facilities to operate

# %%
buildTest.denominati.unique()

# %%
important_amenities = ['school', 'hospital', 'fire_station', 
                       'government', 'toilets', 'courthouse', 
                       'civic', 'place_of_worship', 'museum', 'bank', 
                       'community_centre', 'library', 'convenience', 
                       'bus_station', 'marketplace', 'social_facility', 
                       'clinic', 'university', 'police', 'supermarket', 
                       'college', 'post_office', 'dispatch_center', 
                       'townhall', 'train_station', 'shelter', 'cafe']

# %%
pd.set_option("display.max_columns", None)

# %%
buildingsImpAmen = buildTest.loc[buildTest.amenity.isin(important_amenities)]
buildingsImpAmen['necessaryAccess'] = buildingsImpAmen.amenity
buildingsAC = buildTest.loc[buildTest.air_condit.notna()]
buildingsAC['necessaryAccess'] = 'AC'
buildingsToilets = pd.concat([buildTest.loc[buildTest.toilets.notna()], buildTest.loc[buildTest.toilets_di.notna()]])
buildingsToilets['necessaryAccess'] = 'toilet'
buildingsOutdoorSeating = buildTest.loc[buildTest.outdoor_se == 'yes']
buildingsOutdoorSeating['necessaryAccess'] = 'outdoor_seating'
buildingsInternetAccess = buildTest.loc[buildTest.internet_a.notna()]
buildingsInternetAccess['necessaryAccess'] = 'internet_access'
buildingsSocialServices = pd.concat([buildTest.loc[buildTest.social_fac.notna()], buildTest.loc[buildTest.social_f_1.notna()]])
buildingsSocialServices['necessaryAccess'] = pd.concat([buildTest.social_fac.loc[buildTest.social_fac.notna()], buildTest.social_f_1.loc[buildTest.social_f_1.notna()]])
buildingsHealthcare = pd.concat([buildTest.loc[buildTest.healthcare.notna()], buildTest.loc[buildTest.healthca_1.notna()], buildTest.loc[buildTest.healthca_2.notna()]])
buildingsHealthcare['necessaryAccess'] = pd.concat([buildTest.healthcare.loc[buildTest.healthcare.notna()], buildTest.healthca_1.loc[buildTest.healthca_1.notna()], buildTest.healthca_2.loc[buildTest.healthca_2.notna()]])
buildingsReligion = pd.concat([buildTest.loc[buildTest.religion.notna()], buildTest.loc[buildTest.denominati.notna()]]).drop_duplicates()
buildingsReligion['necessaryAccess'] = 'house_of_worship'

# %%
buildingsNecessary = pd.concat([buildingsImpAmen, buildingsAC, buildingsToilets, buildingsOutdoorSeating, buildingsInternetAccess, buildingsSocialServices, buildingsHealthcare, buildingsReligion]).drop_duplicates().sort_index()

# %%


# ("amenity" IS NOT NULL) OR (("landuse" IS NOT NULL) OR ("shop" IS NOT NULL) OR ("residentia" IS NOT NULL) OR ("government" IS NOT NULL) OR ("office" IS NOT NULL) OR ("tourism" IS NOT NULL) OR ("historic" IS NOT NULL))

for i in inputs:
    if buildTest.amenity[i] is not None:
        continue
    elif buildTest.shop[i] is not None:
        buildTest.amenity[i] = buildTest.shop[i]
    elif buildTest.residentia[i] is not None:
        buildTest.amenity[i] = buildTest.residentia[i]
    elif buildTest.landuse[i] is not None:
        buildTest.amenity[i] = buildTest.landuse[i]
    elif buildTest.tourism[i] is not None:
        buildTest.amenity[i] = buildTest.tourism[i]
    elif buildTest.office[i] is not None:
        buildTest.amenity[i] = buildTest.office[i]
    elif buildTest.historic[i] is not None and buildTest.historic[i] != "yes" and buildTest.building[i] != "vacant":
            buildTest.amenity[i] = buildTest.historic[i]
    elif buildTest.building[i] != "yes" and buildTest.building[i] != "roof":
        buildTest.amenity[i] = buildTest.building[i]
    else:
        buildTest.amenity[i] = "building"


# %%
tess2 = tess.merge(
    buildTest[["amenity", "uID"]], on="uID", how="left"
)

# %%
# AGU Addition - Shannon diversity of building types
shannon_sw = momepy.sw_high(k=3, gdf=tess2, ids='tID')
tess2['bdShDv'] = momepy.Shannon(gdf=tess2, values='amenity', spatial_weights=shannon_sw, 
                              unique_id='tID', categorical=True).series

# %%
tess2.bdShDv[0]

# %%
def applyShannon(chunk_id):
    clls = gpd.read_parquet(folder + f"chunked/chunk_{chunk_id}.pq", columns=characters + ["tessellation", "tID"])
    inputs = iter(range(len(tess2)))
    in2 = iter(range(len(clls)))

    for i in inputs:
        for j in in2:
            if tess2.tID[i] == clls.tID[j]:
                clls.bdShDv[j] = tess2.bdShDv[i]

    
    clls.to_parquet(folder + f"chunked/chunk_{chunk_id}.pq")


# %%
inputs = iter(real_chunks)

for id in inputs:
    applyShannon(id)

# %%
buildTest.amenity

# %%
buildingTypes = buildings.building.copy()

# %%
buildings.amenity.value_counts()

# %%
buildingTypes.unique()

# %%
tags = {'natural': True}
gdf_natural = ox.geometries_from_polygon(polygon, tags)
tags = {'landuse': True}
gdf_landuse = ox.geometries_from_polygon(polygon, tags)
tags = {'leisure': True}
gdf_leisure = ox.geometries_from_polygon(polygon, tags)
tags = {'amenity': True}
gdf_amenity = ox.geometries_from_polygon(polygon, tags)

# %%
buildAmen = buildings.amenity.unique()

# %%
all_amen = gdf_amenity.amenity.unique()

# %%
commonAmen = []
uniqueAmen = []
for amen in all_amen:
    if amen in buildAmen:
        commonAmen.append(amen)
        print(amen + "is in buildings\n")
    else:
        uniqueAmen.append(amen)
        print(amen + "is not in buildings\n")

# %%
commonAmen

# %%
uniqueAmen

# %%
gdf_leisure.leisure.unique()

# %%
gdf_natural.natural.unique()

# %%
gdf_landuse[['landuse', 'geometry']].to_parquet(folder + "osm_landuse.pq")
gdf_landuse.landuse.unique()


# %%
buildingsNecessary.to_parquet(folder + "buildingNecessary.pq")

# %% [markdown]
# Shannon will be calculated based on building type

# %% [markdown]
# 
# ### 2.2. Accessibility
# 
# We use US census data that can be found in the `\Datasets\` folder. The data is originally from US census tabulation blocks for southern california.

# %%
download("https://www2.census.gov/geo/tiger/TIGER2022/TABBLOCK20/tl_2022_06_tabblock20.zip",
        folder + "tiger_blocks", kind="zip")

# %%
#psuedo
# geopandas of population density
# data <- tesselation
# for each tesselation cell intersect with population density shape file, 
# if it is more than one ->
# calculate total area that covers each cell,
# divide that by total area and combine, 
# might also want to do a rough estimate of population per tesselation cell which involves using total people/ha and multiplying by area

popDensity = gpd.read_file("work/data_for_spatial_sig/popDensity/populationDensityData.shp",  enabled_drivers=["ESRI Shapefile"])

# %%
popDensity.head(5)

# %%
data = gpd.read_parquet(folder + "tessellation.pq")

# %%
popArea = data.overlay(popDensity, how='intersection')


# %%
popTemp = popArea.loc[popArea['tID'] == data['tID'][25]]
popTemp.area


# %%
housingTemp = housingData.loc[housingData['tID'] == data['tID'][2]]
housingTemp

# %%
inputs = iter(range(len(data))) #defines iterator that is length of all tesselation cells
totPopDenseList = []
totPopList = []
for i in inputs:
    popTemp = popArea.loc[popArea['tID'] == data['tID'][i]]
    # propArea = promTID.area/sum(promTID.area)
    totPopDense = (popTemp.area/sum(popTemp.area)).dot(popTemp['POP_DENSE'])
    totPop = totPopDense*sum(popTemp.area)*0.0001
    totPopDenseList.append(totPopDense)
    totPopList.append(totPop)

data['pop_dense'] = totPopDenseList
data['population'] = totPopList


# %%
bounds = gpd.GeoSeries(box(*data.total_bounds), crs=data.crs).to_crs(4326)

# %%
data[['tID', 'pop_dense', 'population']].to_parquet(folder + "population/population.pq")

# %%
PopData = pd.read_parquet(folder + "population/population.pq")

# %%
housingData = pd.read_parquet(folder + "housing.pq")

# %%
popHousing = PopData.merge(housingData, on='tID')

# %%
popHousing_gdf = data.merge(popHousing, on='tID')

# %%
highDenseNewTemp = popHousing_gdf.loc[popHousing_gdf['popDenseNew'] == np.max(popHousing_gdf['popDenseNew'])]
nonZeroDenseNewTemp = popHousing_gdf.loc[popHousing_gdf['popDenseNew'] != 0.0]
lowDenseNewTemp = nonZeroDenseNewTemp.loc[nonZeroDenseNewTemp['popDenseNew']  == np.min(nonZeroDenseNewTemp['popDenseNew'])]

# %%
popHousing_gdf['popDenseNew'] = popHousing_gdf.POP20/(popHousing_gdf.area)

# %%
popinfDiff = popHousing.population - popHousing.POP20
popDenseDiff = popHousing.pop_dense - popHousing.population

# %%
np.min(popDenseDiff)

# %%
housingValMin = housingData.loc[housingData.housing_score == np.min(housingData.housing_score)]
housingValMax = housingData.loc[housingData.housing_score == np.max(housingData.housing_score)]

# %%
est_housingExtensive = area_interpolate(geom_data, data, housing, n_jobs=1)

# %%
popHousing.head(1)

# %%
est_housingExtensive.head(1)

# %%
est_housingIntensive.head(1)

# %%
chunks = glob.glob(folder + "chunked/chunk_*.pq")

# %% [markdown]
# The point of these scaling factors is to prevent the difference in population from skewing the data too much. We still want it to have an impact on the access scores, but its important that it doesn't cause too much of a skew since there is a massive amounbt of uncertainty in the population data.

# %%
maxPopScale = np.median(popHousing_gdf['POP20'])#np.percentile(popHousing_gdf['POP20'],75)
DenScale = np.mean(popHousing_gdf['POP20']) + np.std(popHousing_gdf['POP20'])
minPopScale = 1- np.percentile(popHousing_gdf['POP20'],25)/DenScale
#        pop = np.min([minPopScale + pop/DenScale,maxPopScale])


# %%
np.median(popHousing_gdf['POP20'])

# %%
plt.hist(allAccessList, bins=50)

# %%
# Need to rerun this section and use population as it has been interpolated rather than population I calculated. Since there
# seems to be an issue with my computation of population
allConList = []
allBetweenList = []
allStraightList = []
allAccessList = []
for chunk in chunks:
    accessChunk = gpd.read_parquet(chunk)
    inputs = iter(range(len(accessChunk))) #defines iterator that is length of all tesselation cells
    accessScoreList = []
    diversityScore = []
    for i in inputs:
        pop = popHousing_gdf.loc[popHousing_gdf['tID']==accessChunk['tID'][i]]['POP20'].iloc[0]
        pop = np.min([minPopScale + pop/DenScale,maxPopScale])
        connectedNess = accessChunk['clsCnt'][i] * 1000
        betweenNess = accessChunk['btwCnt'][i] * 10
        straightNessScale = (1 + accessChunk['strCnt'][i])
        accessScore = pop * (connectedNess + betweenNess) * straightNessScale
        divScore = pop * accessChunk['bdShDv'][i]
        accessScoreList.append(accessScore)
        diversityScore.append(divScore)
        allConList.append(connectedNess)
        allBetweenList.append(betweenNess)
        allStraightList.append(straightNessScale)
        allAccessList.append(accessScore)
    accessChunk['accessScore'] = accessScoreList
    accessChunk['amenityDiversity'] = diversityScore
    accessChunk[["tID","clsCnt","btwCnt","strCnt", "accessScore", "amenityDiversity"]].to_parquet(folder + f"accessibility/access_{chunk[47:-3]}.pq")
    print(f"Chunk {chunk[47:-3]} processed.")

# %% [markdown]
# ### 2.3. Environmental Prediction Outcomes
# 
# Loading in the environmental prediction outputs from the UMEP model. These were in the form of 100m vector grids, the did not cover the entire study area, but a significant portion.
# 
# We use National Land Cover Database 2016 (https://www.usgs.gov/centers/eros/science/national-land-cover-database?qt-science_center_objects=0#qt-science_center_objects).

# %%
# download("https://s3-us-west-2.amazonaws.com/mrlc/NLCD_2016_Land_Cover_L48_20190424.zip",
#          folder + "land_cover", kind="zip") "work/data_for_spatial_sig/envPredOut/100m_*.shp",  enabled_drivers=["ESRI Shapefile"]

envPredChunks = glob.glob("work/data_for_spatial_sig/envPredOut/100m_*.shp")

# %%
envPredGridTest = gpd.read_file(envPredChunks[0], enabled_drivers=["ESRI Shapefile"]).drop(['GEOID', 'NAME','County_Nam', 'State_Name', 'P0010001', 'ALAND', 'AWATER', 'POP100', 'HU100', 'Shape__Are', 'Shape__Len', 'AREA_HA'], axis=1)
envPredGridTest2 = gpd.read_file(envPredChunks[30], enabled_drivers=["ESRI Shapefile"])


# %%
envPredGridTest3 = gpd.read_file(envPredChunks[31], enabled_drivers=["ESRI Shapefile"])#.drop(['GEOID', 'NAME','County_Nam', 'State_Name', 'P0010001', 'ALAND', 'AWATER', 'POP100', 'HU100', 'Shape__Are', 'Shape__Len', 'AREA_HA'], axis=1)
if 'GEOID' in envPredGridTest3.columns.to_list():
    print("true")
    envPredGridTest3.drop(['GEOID', 'NAME','County_Nam', 'State_Name', 'P0010001', 'ALAND', 'AWATER', 'POP100', 'HU100', 'Shape__Are', 'Shape__Len', 'AREA_HA'], axis=1)

# %%
print(envPredGridTest.columns.to_list())
print(envPredGridTest2.columns.to_list())

# %%
envPredGrid = gpd.read_file(envPredChunks[0], enabled_drivers=["ESRI Shapefile"]).drop(['GEOID', 'NAME','County_Nam', 
                                                                                        'State_Name', 'P0010001', 'ALAND', 
                                                                                        'AWATER', 'POP100', 'HU100', 
                                                                                        'Shape__Are', 'Shape__Len', 
                                                                                        'AREA_HA'], axis=1)

# %% [markdown]
# We open, reproject and clip the data to the extent of tessellation bounds and reproject.

# %%
# lc = xarray.open_rasterio(folder + 'land_cover/NLCD_2016_Land_Cover_L48_20190424.img')
i = 0
envPredGrid = gpd.read_file(envPredChunks[0], enabled_drivers=["ESRI Shapefile"]).drop(['GEOID', 'NAME','County_Nam', 
                                                                                        'State_Name', 'P0010001', 'ALAND', 
                                                                                        'AWATER', 'POP100', 'HU100', 
                                                                                        'Shape__Are', 'Shape__Len', 
                                                                                        'AREA_HA'], axis=1)
for envPredChunk in envPredChunks:
    if i == 0:
        i = 1
        continue
    else:
        envPredDf = gpd.read_file(envPredChunk, enabled_drivers=["ESRI Shapefile"])
        if 'GEOID' in envPredDf.columns.to_list():
            envPredDf = envPredDf.drop(['GEOID', 'NAME','County_Nam', 'State_Name', 'P0010001', 
                            'ALAND', 'AWATER', 'POP100', 'HU100', 'Shape__Are', 
                            'Shape__Len', 'AREA_HA'], axis=1)
        envPredGrid = pd.concat([envPredGrid, envPredDf],ignore_index=True)

# %%
print(envPredGrid.columns.to_list())

# %%
envPredGrid[['left', 'top','right','bottom','POP_DENSE', 'QH', 'QE', 'QS', 'Tsurf', 'Evap', 'RO', 'TotCh', 'Drainage', 'SMD', 'geometry']].to_parquet(folder + "envPred/envPred_grid.pq")

# %% [markdown]
# There is now a need to compute the environmental prediction scores for each grid so as to ascertain which areas are the most at risk vs which ones are the least at risk. We start be normalizing all the values

# %%
def NormalizeVectorGrid(vGrid):
    vectorGrid = vGrid.copy()

    surfaceMoistDeficitArr = vectorGrid['SMD'].copy()
    drainageArr = vectorGrid['Drainage'].copy()
    totMoistureChangeArr = vectorGrid['TotCh'].copy()
    RunOffArr = vectorGrid['RO'].copy()
    evapArr = vectorGrid['Evap'].copy()
    netStorageFluxArr = vectorGrid['QS'].copy()
    latentHeatFluxArr = vectorGrid['QE'].copy()
    sensibleHeatFluxArr = vectorGrid['QH'].copy()
    bulkSurfaceFluxArr = vectorGrid['Tsurf'].copy()

    surfaceMoistDeficitArr = (surfaceMoistDeficitArr-min(surfaceMoistDeficitArr))/(max(surfaceMoistDeficitArr)-min(surfaceMoistDeficitArr))
    drainageArr = (drainageArr-min(drainageArr))/(max(drainageArr) - min(drainageArr))
    totMoistureChangeArr = (totMoistureChangeArr-min(totMoistureChangeArr))/(max(totMoistureChangeArr)-min(totMoistureChangeArr))
    RunOffArr = (RunOffArr-min(RunOffArr))/(max(RunOffArr)-min(RunOffArr))
    evapArr = (evapArr-min(evapArr))/(max(evapArr)-min(evapArr))
    netStorageFluxArr = (netStorageFluxArr-min(netStorageFluxArr))/(max(netStorageFluxArr)-min(netStorageFluxArr))
    latentHeatFluxArr = (latentHeatFluxArr-min(latentHeatFluxArr))/(max(latentHeatFluxArr)-min(latentHeatFluxArr))
    sensibleHeatFluxArr = (sensibleHeatFluxArr-min(sensibleHeatFluxArr))/(max(sensibleHeatFluxArr)-min(sensibleHeatFluxArr))
    bulkSurfaceFluxArr = (bulkSurfaceFluxArr-min(bulkSurfaceFluxArr))/(max(bulkSurfaceFluxArr)-min(bulkSurfaceFluxArr))

    vectorGrid['SMD_norm'] = surfaceMoistDeficitArr
    vectorGrid['Drainage_norm'] = drainageArr
    vectorGrid['TotCh_norm'] = totMoistureChangeArr
    vectorGrid['RO_norm'] = RunOffArr
    vectorGrid['Evap_norm'] = evapArr
    vectorGrid['QS_norm'] = netStorageFluxArr
    vectorGrid['QE_norm'] = latentHeatFluxArr
    vectorGrid['QH_norm'] = sensibleHeatFluxArr
    vectorGrid['Tsurf_norm'] = bulkSurfaceFluxArr

    return vectorGrid


# %% [markdown]
# After the grid cells have all been normalized we compute which grid cell is predicted to have the worst environmental outcomes, which we determine based on which areas store the most energy (minimal energy flux) and store the least amount of water (high water flux). However not all water fluxes are bad, if an area has high evaporation, that means there is a high rate of evapotranspiration and thus the score is considered higher. 

# %%
def scoreCells(vectGrid, norm=True):#, num=1, best=True, worst=True):
    # if not normalized ensure that the grid is normalized
    if not norm:
        vectGrid = NormalizeVectorGrid(vectGrid)

    surfaceMoistDeficitArr = vectGrid['SMD_norm'].copy() #lower Better
    surfaceMoistDeficitArr = surfaceMoistDeficitArr * -1 + 1
    # drainageArr = vectGrid['Drainage_norm'].copy() # lower better
    # drainageArr = drainageArr * -1 + 1 # This needs to be rediscussed -> might be better to scale by half or use it in conjunction with runoff somehow? -> should remove drainage since it is already considered in runoff and soil moisture state
    totMoistureChangeArr = vectGrid['TotCh_norm'].copy() # higher better
    RunOffArr = vectGrid['RO_norm'].copy() # lower better
    RunOffArr = RunOffArr * -1 + 1
    evapArr = vectGrid['Evap_norm'].copy() # lower better for water balance, higher better for energy balance (more evapotranspiration)
    evapArrWater = evapArr * -1 + 1
    netStorageFluxArr = vectGrid['QS_norm'].copy() # lower better
    netStorageFluxArr = netStorageFluxArr * -1 + 1
    latentHeatFluxArr = vectGrid['QE_norm'].copy() # higher better
    sensibleHeatFluxArr = vectGrid['QH_norm'].copy() # lower better
    sensibleHeatFluxArr = sensibleHeatFluxArr * -1 + 1
    bulkSurfaceFluxArr = vectGrid['Tsurf_norm'].copy() # lower better
    bulkSurfaceFluxArr = bulkSurfaceFluxArr * -1 + 1

    heat_score = netStorageFluxArr + latentHeatFluxArr + sensibleHeatFluxArr + bulkSurfaceFluxArr + evapArr
    water_score = surfaceMoistDeficitArr + totMoistureChangeArr + RunOffArr + evapArrWater

    scoreGrid = vectGrid.copy()
    scoreGrid["heat_score"] = heat_score
    scoreGrid["water_score"] = water_score
    scoreGrid = scoreGrid.drop(columns=["SMD_norm","Drainage_norm","TotCh_norm","RO_norm","Evap_norm","QS_norm","QE_norm","QH_norm","Tsurf_norm"])
    return scoreGrid

# %%
envPred = gpd.read_parquet(folder + "envPred/envPred_grid.pq")
envPredNorm = NormalizeVectorGrid(envPred)
envPredScore = scoreCells(envPredNorm)

# %%
envPredScore.to_parquet(folder + "envPredGridScored")
envPredScore.head(2)

# %% [markdown]
# ### 2.4 Workplace area characteristics
# 
# We use [LEHD Origin-Destination Employment Statistics (LODES)](https://lehd.ces.census.gov/data/lodes/LODES7/LODESTechDoc7.5.pdf).
# 
# For that we also need official block geometries.

# %%
download("https://www2.census.gov/geo/tiger/TIGER2022/TABBLOCK20/tl_2022_06_tabblock20.zip",
        folder + "tiger_blocks", kind="zip")

# %%
geoms = gpd.read_file(folder + 'tiger_blocks') 

# %%
geoms

# %%
geoms.GEOID20 = geoms.GEOID20.astype("int64")

# %%
wac = pd.read_csv("https://lehd.ces.census.gov/data/lodes/LODES7/ca/wac/ca_wac_S000_JT00_2019.csv.gz")

# %%
wac

# %% [markdown]
# Once we have data and geometries, we can join them together.

# %%
merged = geoms.merge(wac, left_on="GEOID20", right_on="w_geocode", how="left").fillna(0)

# %%
merged.shape

# %% [markdown]
# We are interested only in some columns.

# %%
jobs = ['C000', #Total number of jobs
        'CE01', #Number of jobs with earnings $1250/month or less
        'CE02', #Number of jobs with earnings $1251/month to $3333/month
        'CE03', #Number of jobs with earnings greater than $3333/month
        'CD01', #Number of jobs for workers with Educational Attainment: Less than high school
        'CD02', #Number of jobs for workers with Educational Attainment: High school or equivalent, no college
        'CD03', #Number of jobs for workers with Educational Attainment: Some college or Associate degree
        'CD04', #Number of jobs for workers with Educational Attainment: Bachelor's degree or advanced degree
        'createdate'] #Date on which data was created, formatted as YYYYMMDD 

# %%
housing = ['HOUSING20', #Housing count from the 2020 Census
        'POP20'] #Population count from the 2020 Census

# %%
geom_data = geoms.copy()

# %%
geom_data["housing_score"] = geom_data.HOUSING20 - geom_data.POP20
geom_data = geom_data.to_crs(data.crs)

# %%
housingValOrigMin = geom_data.loc[geom_data.housing_score == np.min(geom_data.housing_score)]
housingValOrigMax = geom_data.loc[geom_data.housing_score == np.max(geom_data.housing_score)]

# %%
housingValOrigMax

# %%
merged = merged.to_crs(data.crs)

# %% [markdown]
# Now we will transfer the selected data to tessallation using area weighted interpolation.

# %%
%time est = area_interpolate(merged, data, jobs, n_jobs=1)

# %%
%time est_housing = area_interpolate(geom_data, data, housing, n_jobs=1)

# %%
est['tID'] = data['tID'].values

# %%
est_housing['tID'] = data['tID'].values

# %%
est_housing.drop(columns="geometry").to_parquet(folder + "housing.pq")

# %%
est.drop(columns="geometry").to_parquet(folder + "workplace.pq")

# %%
housing_score = est_housing.HOUSING20 - est_housing.POP20

# %%
est_housing['housing_score'] = housing_score

# %% [markdown]
# ### 2.5 NDVI
# 
# Normalized difference vegetation index (NDVI) is calculated using Sentinel 2 data. 
# 
# First we have to download and reproject mosaic. We specify the local directory and use GeoJSON with information on tiles to get a correct tile.

# %%
losAngeles = gpd.read_parquet(folder + "enclosures.pq") # defining area of interest

# %%
losAngeles_extent = gpd.GeoSeries(box(*losAngeles.total_bounds), crs=losAngeles.crs).to_crs(tess.crs).iloc[0]

# %% [markdown]
# Now we know which tile we need and its specification.

# %% [markdown]
# We can then automatically download and the scene.

# %% [markdown]
# Since the study area crosses multiple tiles, we open each of then, clip it and merge them together.

# %% [markdown]
# Now we compute the actual NDVI in a loop for each chunk.

# %%
r = rioxarray.open_rasterio(folder + "sentinel2_allBands.tif")
clipped = r.rio.clip_box(*losAngeles.total_bounds)
masked = clipped.where(clipped != 0)
ndvi = (masked.sel(band=8) - masked.sel(band=4)) / (masked.sel(band=8) + masked.sel(band=4))

# %%
chunks = glob.glob(folder + "chunked/chunk_*.pq")

# %%
for chunk in chunks:
    losAngeles = gpd.read_parquet(chunk, columns=["tID", "tessellation"])
    r = rioxarray.open_rasterio(folder + "sentinel2_allBands.tif")
    clipped = r.rio.clip_box(*losAngeles.total_bounds)
    masked = clipped.where(clipped != 0)
    ndvi = (masked.sel(band=8) - masked.sel(band=4)) / (masked.sel(band=8) + masked.sel(band=4))
    affine = ndvi.rio.transform()
    stats = rasterstats.zonal_stats(
        losAngeles.geometry, 
        raster=ndvi.values,
        affine=affine,
        stats=['mean'],
        all_touched=True,
        nodata = np.nan,
    )
    losAngeles['ndvi'] = [x["mean"] for x in stats]
    losAngeles[["tID", "ndvi"]].to_parquet(folder + f"NDVI/NDVI{chunk[47:-3]}.pq")
    print(f"Chunk {chunk[47:-3]} processed.")

# %% [markdown]
# ### 2.6 Buildings that are neccessary to reach
# 
# We determined the buildings and spaces that are valuable to reach within 15 minutes of walking based on literature. These were already determined in a previous step.

# %% [markdown]
# Since we have data fro the whole US, lets open it with a mask of our case study area.

# %%
enclosures = gpd.read_parquet(folder + "enclosures.pq")

# %%
buildingsNecessary = gpd.read_parquet(folder + "buildingNecessary.pq")

# %%
buildingsNecessary = buildingsNecessary.to_crs(enclosures.crs)

# %% [markdown]
# We loop over chunks, create pandana network and measure the number of historical buildings in 15 minutes walking distance.

# %%
cell0 = gpd.read_parquet(folder + f'chunked/chunk_0.pq')


# %%
buildingsNecessary_c

# %%
listBuilding = buildingsNecessary.columns.drop('geometry').to_list()
buildingsNecessary=buildingsNecessary.drop_duplicates('uID')

# %%
buildingsNecessaryCentroids = gpd.GeoDataFrame(buildingsNecessary[listBuilding], geometry=buildingsNecessary.centroid)

buildingsNecessaryCentroids.head(5)

# %%
for c in real_chunks:    
    nodes = gpd.read_parquet(folder + f'nodes/nodes_{c}.pq')
    edges = gpd.read_parquet(folder + f'edges/edges_{c}.pq')
    # nodes = nodes.set_index('nodeID')
    network = pandana.Network(nodes.geometry.x, nodes.geometry.y, 
                          edges['node_start'], edges['node_end'], edges[['mm_len']])
    network.precompute(1200)

    buildingsNecessary_c = gpd.clip(buildingsNecessaryCentroids, box(*nodes.total_bounds))
    if len(buildingsNecessary_c) > 0:
        network.set_pois(category = 'necessary_buildings',
                         maxdist = 1200,
                         maxitems=500,
                         x_col = buildingsNecessary_c.geometry.x, 
                         y_col = buildingsNecessary_c.geometry.y)

        res = network.nearest_pois(distance = 1200,
                                   category = 'necessary_buildings',
                                   num_pois = len(buildingsNecessary_c),
                                   include_poi_ids = False)

        res = res.replace(1200, pd.NA)
        nodes['buildingNecessary_counts'] = res.count(axis=1)
    else:
        nodes['buildingNecessary_counts'] = pd.NA
    
    cells = gpd.read_parquet(folder + f'chunked/chunk_{c}.pq')
    cells['edgeID'] = get_edge_ratios(cells, edges)
    cells['nodeID'] = get_nodes(cell0, nodes, edges, 'nodeID', 'edgeID', 'node_start', 'node_end')
    cells = cells.merge(nodes, on='nodeID', how='left')
    cells[['tID', 'buildingNecessary_counts']].to_parquet(folder + f"NecessaryBuildings/necBuild_{c}.pq")
    print(f"chunk {c} processed")

# %% [markdown]
# ## 3. Cluster analysis
# 
# Cluster analysis uses the data on form, function, and their combination in a K-Means clustering to identify homogenous patterns of built form. 
# TODO:
# - Need to define bounds
# - Select for spatial signatures that one would expect a city planner to have a say in 
# - Find clusters within new bounds, and assign scoring (both with and without amenity closeness)
# - Interpolate new qualities for all this to define policies, then run SMOO

# %% [markdown]
# ### 3.1. Form
# 
# Clustering based on form uses contextualised characters as an input. We read all chunks at once with dask and get a single pandas dataframe for clustering.
# Need to interpolate for the shapes defined by the environmental prediction model.

# %% [markdown]
# #### 3.1.1. Data Processing 
# 
# Need to prepare all the data for clustering analysis

# %%
used_characteristics_v2 = {
    # 'sdbCoA', # 'courtyard area of building',
    'sscCCo', # 'circular compactness of ETC', #
    # 'stcOri', # 'orientation of ETC', #
    'sicCAR', # 'covered area ratio of ETC', # building cover
    # 'stbCeA', # 'cell alignment of building',
    'mtbAli', # 'alignment of neighbouring buildings',
    # 'mtbNDi', # 'mean distance between neighbouring buildings',
    # 'mtcWNe', # 'perimeter-weighted neighbours of ETC', 
    # 'ltcWRE', # 'weighted reached enclosures of ETC',
    'ltbIBD', # 'mean inter-building distance',
    # 'sdsSPW', # 'width of street profile',
    # 'sdsSWD', # 'width deviation of street profile',
    'sdsSPO', # 'openness of street profile', #
    # 'sssLin', # 'linearity of street segment', #
    # 'ldsMSL', # 'mean segment length within 3 steps',
    # 'mtdDeg', # 'node degree of junction',
    # 'lcdMes', # 'local meshedness of street network',
    # 'lcnClo', # 'local closeness of street network',
    # 'xcnSCl', # 'square clustering of street network',
    # 'mtdMDi', # 'mean distance to neighbouring nodes of street network',
    'lddNDe', # 'local node density of street network',
    # 'linWID', # 'local degree weighted node density of street network',
    # 'stcSAl', # 'street alignment of ETC', 
    # 'sddAre', # 'area covered by node-attached ETCs',
    # 'sdsAre', # 'area covered by edge-attached ETCs',
    # 'sisBpM', # 'buildings per meter of street segment',
    # 'misCel', # 'reached ETCs by neighbouring segments',
    # 'mdsAre', # 'reached area by neighbouring segments',
    # 'lisCel', # 'reached ETCs by local street network',
    # 'ldsAre', # 'reached area by local street network',
    # 'ltcRea', # 'reached ETCs by tessellation contiguity',
    # 'ldePer', # 'perimeter of enclosure',
    # 'lseCCo', # 'circular compactness of enclosure',
    # 'lseERI', # 'equivalent rectangular index of enclosure',
    # 'lseCWA', # 'compactness-weighted axis of enclosure',
    # 'lteOri', # 'orientation of enclosure',
    # 'lteWNB', # 'perimeter-weighted neighbours of enclosure',
    'lieWCe', # 'area-weighted ETCs of enclosure', # 
}

# %%
used_characteristics = {
    'sdbCoA', # 'courtyard area of building',
    'sscCCo', # 'circular compactness of ETC', #
    'stcOri', # 'orientation of ETC', #
    'sicCAR', # 'covered area ratio of ETC', # building cover
    'stbCeA', # 'cell alignment of building',
    'mtbAli', # 'alignment of neighbouring buildings',
    'mtbNDi', # 'mean distance between neighbouring buildings',
    'mtcWNe', # 'perimeter-weighted neighbours of ETC', 
    'ltcWRE', # 'weighted reached enclosures of ETC',
    'ltbIBD', # 'mean inter-building distance',
    'sdsSPW', # 'width of street profile',
    'sdsSWD', # 'width deviation of street profile',
    'sdsSPO', # 'openness of street profile', #
    'sssLin', # 'linearity of street segment', #
    'ldsMSL', # 'mean segment length within 3 steps',
    'mtdDeg', # 'node degree of junction',
    'lcdMes', # 'local meshedness of street network',
    'lcnClo', # 'local closeness of street network',
    'xcnSCl', # 'square clustering of street network',
    'mtdMDi', # 'mean distance to neighbouring nodes of street network',
    'lddNDe', # 'local node density of street network',
    'linWID', # 'local degree weighted node density of street network',
    # 'stcSAl', # 'street alignment of ETC', 
    # 'sddAre', # 'area covered by node-attached ETCs',
    'sdsAre', # 'area covered by edge-attached ETCs',
    'sisBpM', # 'buildings per meter of street segment',
    'misCel', # 'reached ETCs by neighbouring segments',
    'mdsAre', # 'reached area by neighbouring segments',
    'lisCel', # 'reached ETCs by local street network',
    'ldsAre', # 'reached area by local street network',
    'ltcRea', # 'reached ETCs by tessellation contiguity',
    'ldePer', # 'perimeter of enclosure',
    'lseCCo', # 'circular compactness of enclosure',
    'lseERI', # 'equivalent rectangular index of enclosure',
    'lseCWA', # 'compactness-weighted axis of enclosure',
    'lteOri', # 'orientation of enclosure',
    'lteWNB', # 'perimeter-weighted neighbours of enclosure',
    'lieWCe', # 'area-weighted ETCs of enclosure', # 
}

# %%
usedCharForm = []
for char in used_characteristics:
    usedCharForm.append(char + "_q1")
    usedCharForm.append(char + "_q2")
    usedCharForm.append(char + "_q3")

# %%
usedCharForm2 = []
for char in used_characteristics_v2:
    usedCharForm2.append(char + "_q1")
    usedCharForm2.append(char + "_q2")
    usedCharForm2.append(char + "_q3")

# %%
chunks_all = dd.read_parquet(folder + "chunked/*")
chunks_all = chunks_all.compute()

# %%
chunks_all = chunks_all.set_index('tID').sort_index()
chunks_all

# %%
ChunksAoI = chunks_all.iloc[tessEnvPredArea.tID.to_list()]
ChunksAoI = ChunksAoI.replace(np.inf, np.nan)
ChunksAoI_gdf = gpd.GeoDataFrame(data=ChunksAoI, geometry=tessEnvPredArea.tessellation)

# %%
ChunksAoI_gdf.drop(columns=["tessellation","buildings"], inplace=True)

# %%
ChunksAoI_gdf.to_parquet(folder + "combTessChunks.pq")

# %%
form = dd.read_parquet(folder + "convolutions/*")

# %%
form = form.compute()

# %% [markdown]
# Find the tesselation cells within the boundaries than cutoff the form based off those vals then do the clusters

# %%
form = form.sort_index()
form

# %%
tess = gpd.read_parquet(folder + "tess.pq", columns=["tID", "buildings", "tessellation"])

# %%
fullTess = gpd.read_parquet(folder + "tess.pq")

# %%
fullTess.head(1)

# %%
envPred = gpd.read_parquet(folder + "envPred/envPred_grid.pq")

# %%
tessEnvPredArea = gpd.clip(tess, box(*envPred.total_bounds))
tessEnvPredArea = tessEnvPredArea.sort_index()

# %%
form

# %%
formAoI = form.iloc[tessEnvPredArea.tID.to_list()]

# %%
formAoIFull = form.iloc[tessEnvPredArea.tID.to_list()]
formAoIFull = formAoIFull[formAoIFull.columns.intersection(usedCharForm)]

# %%
formAoITest2 = formAoI[formAoI.columns.intersection(usedCharForm2)]

# %%
formAoI = formAoITest2

# %%
formAoI.head(1)

# %%
#formAoI = formAoI.drop(columns=['sddAre_q1','sddAre_q2', 'sddAre_q3'])  # bug in the code above, should not matter much

# %%
formAoI = formAoI.replace(np.inf, np.nan)

# %%
formAoI_gdf = gpd.GeoDataFrame(data=formAoI, geometry=tessEnvPredArea.tessellation)

# %%
formAoIFull_gdf = gpd.GeoDataFrame(data=formAoIFull, geometry=tessEnvPredArea.tessellation)

# %%
chunks_all_AoI = chunks_all.iloc[tessEnvPredArea.tID.to_list()]
chunksAll_gdf = gpd.GeoDataFrame(data=chunks_all_AoI, geometry=tessEnvPredArea.tessellation)

# %%
chunksAll_gdf.to_parquet(folder + "combChunks.pq")

# %%
formAoI_gdf.to_parquet(folder + "convolutions_AoI_usedCharsFinal.pq")

# %% [markdown]
# ##### Old data preparing

# %%
formAoIFull_gdf.to_parquet(folder + "convolutions_AoI_usedCharsFullFinal.pq")

# %%
spatialSignaturesTrends = envPred.copy()

# %%
spatialSignaturesOpt = envPred.copy()

# %%
spatialSignaturesTrends = area_interpolate(formAoI_gdf, spatialSignaturesTrends, usedCharForm2, n_jobs=1)


# %%
spatialSignaturesOpt = area_interpolate(chunksAll_gdf, spatialSignaturesOpt, list(used_characteristics_v2), n_jobs=1)


# %% [markdown]
# Raw data needs to be standardised. We use standard scaler.

# %%
spatialSignaturesOpt.to_parquet(folder + "spatialSignatureGrid_forOpt2.pq")

# %%
ss_trends_df = pd.DataFrame(spatialSignaturesTrends.drop(columns=['geometry']))

# %%
scalerGrid = preprocessing.StandardScaler()
dataGrid = scalerGrid.fit_transform(ss_trends_df)

# %% [markdown]
# Due to tiny minor topology issue, we have a few missing values in the whole datasets so we replace them with 0. In a dataset of 2 million rows, it does not matter much.

# %% [markdown]
# ##### Current Preprocessing

# %%
formAoI_gdf = gpd.read_parquet(folder + "convolutions_AoI_usedCharsFinal.pq")

# %%
formAoI_gdf.head(1)

# %%
formAoI = formAoI_gdf.drop(columns=["geometry"])

# %%
scaler = preprocessing.StandardScaler()
data = scaler.fit_transform(formAoI)
data[np.isnan(data)] = 0

# %% [markdown]
# ##### Old data fitting

# %%
scaler = preprocessing.StandardScaler()
dataFull = scaler.fit_transform(formAoIFull)
dataFull[np.isnan(dataFull)] = 0

# %%
dataGrid[np.isnan(dataGrid)] = 0

# %% [markdown]
# #### 3.1.2. Clustering of Form
# 
# We use clustergram to determine the optimal number of clusters, using 10 initializations within range from 1 to 30 clusters.

# %% [markdown]
# ##### Old clustering attempts

# %%
cg_formFull = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_formFull.fit(dataFull)


ax1 = cg_formFull.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial Signatures set Full Silhouette score")

# %%
cg_formFull_attempt2 = Clustergram(range(1, 30), backend='sklearn', n_init=20)
cg_formFull_attempt2.fit(dataFull)


ax1 = cg_formFull_attempt2.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial Signatures set 2 Silhouette score")

# %% [markdown]
# ##### Current clustering attempt

# %%
cg_form = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_form.fit(data)


ax1 = cg_form.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial Signatures Silhouette score")

# %%
ax1 = cg_form.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial Signatures Silhouette score")
c = ugg.COLORS
# ax1.axvline(4, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(11, color=c[4], ymin=.02, ymax=.98)
# ax1.axvline(22, color=c[4], ymin=.02, ymax=.98)


# %% [markdown]
# From the above results it can be seen that given the scale of the information (namely only having 2999 ETCs), the dimensionality of the data is not a helping factor, in fact it can act as a limiting factor, where the greater data causes less ability for the centroids of the clusters to be discernable, as seen from the Silhouette coefficient. From the above Silhouette scores, local peaks can be seen at 4 clusters, 11 clusters, and 22 clusters (looking at the first set of Silhouette scores not the second).

# %% [markdown]
# ##### Second set of old clustering attempts

# %%
cg_form_attempt2 = Clustergram(range(1, 30), backend='sklearn', n_init=20)
cg_form_attempt2.fit(data)


ax1 = cg_form_attempt2.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial Signatures set 2 Silhouette score")

# %%
ax1 = cg_form_attempt2.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial Signatures set 2 Silhouette score")
ax1.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(11, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(17, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# ##### Tesselation (again)
# Now we can plot resulting clustergam. Due to outliers, we truncate y axis to see the behaviour of clusters.

# %%
ax = cg_form.plot(figsize=(20, 20), linewidth=0.5, cluster_style={"edgecolor": "r", "alpha":.6}, size=1,
       line_style={"alpha":.5})
# ax.set_ylim(-10, 40)

# %% [markdown]
# We can see from the clustergram that the areas post major splitting occurs around the same points as where the peaks of the Silhouette scores were, namely at 4 clusters, 11 clusters, and 22 clusters. However from a qualitative analysis of the above graph, 6 and 7 clusters also illustrate a potential area of interest. However, it must be stated that given the homogeneity of urban form within Los Angeles, the expectation for strong Silhouette scores might have been unreasonable.

# %%
with open(folder + 'clustergram_form_Final.pickle','wb') as f:
    pickle.dump(cg_form, f)

# %% [markdown]
# ##### Grid (bleh)

# %%
ax = cg_form_grid.plot(figsize=(20, 20), linewidth=0.5, cluster_style={"edgecolor": "r", "alpha":.6}, size=1,
       line_style={"alpha":.5})
# ax.set_ylim(-10, 40)

# %% [markdown]
# 8, 15, or 20 for the grid clusters

# %%
with open(folder + 'clustergram_form_grid2.pickle','wb') as f:
    pickle.dump(cg_form_grid, f)

# %% [markdown]
# ##### Kmeans clustering tesselation cells final set

# %%
%time km_form1 = KMeans(n_clusters=4, n_init=100).fit(data)
%time km_form2 = KMeans(n_clusters=6, n_init=100).fit(data)
%time km_form3 = KMeans(n_clusters=7, n_init=100).fit(data)
%time km_form4 = KMeans(n_clusters=11, n_init=100).fit(data)
%time km_form5 = KMeans(n_clusters=22, n_init=100).fit(data)

# %%
km_form = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(data)


# %%
tessSpatialData = gpd.read_parquet(folder + "tesselation_form_clustering_final.pq")

# %%
tessSpatialData.drop(columns=["clusters_form_4", "clusters_form_6", "clusters_form_7","clusters_form_11","clusters_form_22"], inplace=True)
tessSpatialData.head(1)

# %%
tessSpatialData = tessSpatialData.set_index('tID').sort_index()

# %%
tessSpatialData["clusters_spatial"] = km_form.labels_

# %%
tessSpatialData["clusters_form_4"] = km_form1.labels_
tessSpatialData["clusters_form_6"] = km_form2.labels_
tessSpatialData["clusters_form_7"] = km_form3.labels_
tessSpatialData["clusters_form_11"] = km_form4.labels_
tessSpatialData["clusters_form_22"] = km_form5.labels_

# %%
ax = tessSpatialData.plot('clusters_spatial', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial Form Clusters ETCs")
ax.set_axis_off()
ax = tessSpatialData.set_geometry('buildings').plot('clusters_spatial', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial Form Clusters Buildings")
ax.set_axis_off()

# %%
ax = tessSpatialData.plot('clusters_form_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial Form Clusters ETCs (4 clusters)")
ax.set_axis_off()
ax = tessSpatialData.plot('clusters_form_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial Form Clusters ETCs (6 clusters)")
ax.set_axis_off()
ax = tessSpatialData.plot('clusters_form_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial Form Clusters ETCs (7 clusters)")
ax.set_axis_off()
ax = tessSpatialData.plot('clusters_form_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial Form Clusters ETCs (11 clusters)")
ax.set_axis_off()
ax = tessSpatialData.plot('clusters_form_22', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial Form Clusters ETCs (22 clusters)")
ax.set_axis_off()

# %%
ax = tessSpatialData.set_geometry('buildings').plot('clusters_form_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial Form Clusters Buildings (4 clusters)")
ax.set_axis_off()
ax = tessSpatialData.set_geometry('buildings').plot('clusters_form_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial Form Clusters Buildings (6 clusters)")
ax.set_axis_off()
ax = tessSpatialData.set_geometry('buildings').plot('clusters_form_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial Form Clusters Buildings (7 clusters)")
ax.set_axis_off()
ax = tessSpatialData.set_geometry('buildings').plot('clusters_form_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial Form Clusters Buildings (11 clusters)")
ax.set_axis_off()
ax = tessSpatialData.set_geometry('buildings').plot('clusters_form_22', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial Form Clusters Buildings (22 clusters)")
ax.set_axis_off()

# %% [markdown]
# It can be seen that the most interesting cluster of form is at 7 and 11 clusters, since then the differences between the structures seem to split the city into distinct parts with common traits that can be compared to other aspects later.

# %%
tessSpatialData.to_parquet(folder + "tesselation_form_clustering_final.pq")

# %% [markdown]
# ##### Old Data (no longer relevant)

# %%
%time km_form = KMeans(n_clusters=16, n_init=100).fit(data)

# %%
%time km_form_grid = KMeans(n_clusters=8, n_init=100).fit(dataGrid)

# %%
%time km_form2 = KMeans(n_clusters=13, n_init=100).fit(data)

# %%
# %time km_form_grid2 = KMeans(n_clusters=15, n_init=100).fit(dataGrid)
%time km_form_grid2 = KMeans(n_clusters=13, n_init=100).fit(dataGrid)

# %%
%time km_form3 = KMeans(n_clusters=10, n_init=100).fit(data)

# %%
%time km_form_grid3 = KMeans(n_clusters=20, n_init=100).fit(dataGrid)

# %%
tessEnvPredArea['clusters_form_16'] = km_form.labels_
tessEnvPredArea['clusters_form_13'] = km_form2.labels_
tessEnvPredArea['clusters_form_10'] = km_form3.labels_

# %%
tessEnvPredArea.head(1)

# %%
spatialSignaturesTrends['clusters_form_8'] = km_form_grid.labels_
spatialSignaturesTrends['clusters_form_13'] = km_form_grid2.labels_
spatialSignaturesTrends['clusters_form_20'] = km_form_grid3.labels_

# %%
spatialSignaturesOpt['clusters_form'] = km_form_grid2.labels_

# %%
constraintsTemp

# %%
cellNums = iter(range(len(spatialSignaturesOpt)))
usedNums = []
# constraints = spatialSignaturesOpt.head(1).copy()
constraints = spatialSignaturesOpt.head(1).copy()
# constraints["ranges"] = ["mean", "max", "min"]
# constraints = constraints.set_index("ranges")
# clusterRange = list(range(max(spatialSignaturesOpt.clusters_form)+1))

constraintsFull = pd.DataFrame()

for num in cellNums:
    if spatialSignaturesOpt.clusters_form[num] in usedNums:
        continue
    else:
        clust_num = spatialSignaturesOpt.clusters_form[num]
        constraintsTemp = constraints.copy().drop(columns=['geometry'])
        usedNums.append(clust_num)
        spatialCluster = spatialSignaturesOpt.loc[spatialSignaturesOpt.clusters_form == clust_num]
        for col in spatialSignaturesOpt.columns:
            if col != "geometry":
                constraintsTemp[col] = [[np.mean(spatialCluster[col]), np.max(spatialCluster[col]), np.min(spatialCluster[col])]]
                # constraintsTemp[col] = np.max(spatialCluster[col])
                # constraintsTemp[col] = np.min(spatialCluster[col])
        # constraintsTemp = constraintsTemp.set_index(clust_num)
        constraintsTemp.clusters_form = clust_num
        constraintsFull = pd.concat([constraintsTemp,constraintsFull])
constraints = constraintsFull.set_index('clusters_form').sort_index()

# %%
constraints.to_parquet(folder + "constraints_per_Cluster.pq")

# %% [markdown]
# We can plot the labels based on tessellation or based on buildings, each providing different insights.

# %%
ax = tessEnvPredArea.plot('clusters_form_16', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial Form Clusters (16 clusters)")
ax.set_axis_off()
ax = tessEnvPredArea.plot('clusters_form_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial Form Clusters (13 clusters)")
ax.set_axis_off()
ax = tessEnvPredArea.plot('clusters_form_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial Form Clusters (10 clusters)")
ax.set_axis_off()

# %% [markdown]
# From the above plots it seems that 13 clusters is the sweet spot for clustering based on the chosen spatial aspects.

# %% [markdown]
# ##### Grids - not useful other than demonstrating this type of interpolation causes dimensionality to be a bigger issue

# %%
ax = spatialSignaturesTrends.plot('clusters_form_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()

# %%
ax = spatialSignaturesTrends.plot('clusters_form_20', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()

# %% [markdown]
# ##### Tesselations (again)

# %%
ax = tessEnvPredArea.set_geometry('buildings').plot('clusters_form', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()

# %%
ax = tessEnvPredArea_v2.plot('clusters_form', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()

# %%
ax = tessEnvPredArea_v2.set_geometry('buildings').plot('clusters_form', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()

# %%
ax = tessEnvPredArea_v3.plot('clusters_form', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()

# %%
ax = tessEnvPredArea_v3.set_geometry('buildings').plot('clusters_form', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()

# %%
tessEnvPredArea.to_parquet(folder + 'tess_AoI_clusterForm_newChars.pq')
# tessEnvPredArea_v2.to_parquet(folder + 'tess_AoI_clusterForm_12.pq')
# tessEnvPredArea_v3.to_parquet(folder + 'tess_AoI_clusterForm_22.pq')

# %%
tessEnvPredArea = gpd.read_parquet(folder + 'tess_AoI_clusterForm_newChars.pq')

# %%
tessEnvPredArea.head(1)

# %%
spatialSignaturesTrends.to_parquet(folder + 'spatialSignaturesGrid2.pq')

# %%
spatialSignaturesTrends1 = gpd.read_parquet(folder + 'spatialSignaturesGrid.pq')

# %%
spatialSignaturesTrends1.head(1)

# %%
spatialSignaturesTrends2 = gpd.read_parquet(folder + 'spatialSignaturesGrid2.pq')

# %%
spatialSignaturesTrends2.head(1)

# %% [markdown]
# ### 3.2. Social Function Clustering
# 
# Clustering based on function uses only data on function, which we need to prepare. First we have to merge all together.

# %%
housing = pd.read_parquet(folder + "housing.pq")
housingTess = tess.merge(housing, on="tID")


# %%
housingTess.to_parquet(folder + "housingTess.pq")

# %%
housingTessAoI = housingTess.iloc[tessEnvPredArea.tID.to_list()]
housingTessAoI.to_parquet(folder + "housingTessAoI.pq")

# %%
population = pd.read_parquet(folder + "population/population.pq")
populationAoI = population.iloc[tessEnvPredArea.tID.to_list()]


# %%
envPred = gpd.read_parquet(folder + "envPred/envPred_grid.pq")

# %% [markdown]
# ##### Interesting but out of scope data
# The workplace data is interesting and indicates something on social stratification, however it is beyond the scope of this project.

# %%
workplace = pd.read_parquet(folder + "workplace.pq")
workplaceAoI = workplace.iloc[tessAccess.tID.to_list()] # defined later but simpler for me


# %%
workplaceGDF = gpd.GeoDataFrame(data=workplaceAoI, geometry=tessAccess.tessellation)

# %%
workplaceGDF.to_parquet(folder + "workplace_tess.pq")

# %%
jobs = ['C000', #Total number of jobs
        'CE01', #Number of jobs with earnings $1250/month or less
        'CE02', #Number of jobs with earnings $1251/month to $3333/month
        'CE03', #Number of jobs with earnings greater than $3333/month
        'CD01', #Number of jobs for workers with Educational Attainment: Less than high school
        'CD02', #Number of jobs for workers with Educational Attainment: High school or equivalent, no college
        'CD03', #Number of jobs for workers with Educational Attainment: Some college or Associate degree
        'CD04', #Number of jobs for workers with Educational Attainment: Bachelor's degree or advanced degree
        'createdate'] #Date on which data was created, formatted as YYYYMMDD 

# %%
workplaceAoI.head(1)

# %% [markdown]
# ##### Accessibility 

# %%
chunksCobbled = dd.read_parquet(folder + "chunked/chunk_*.pq").compute()

# %%
np.max(chunksCobbled.btwCnt)

# %%
np.median(chunksCobbled.clsCnt)

# %%
accessibility = dd.read_parquet(folder + "accessibility/*.pq").compute()


# %%
accessibility

# %%
tess.head(1)

# %%
tessAccessFull = tess.merge(accessibility, on="tID")
tessAccess = tessAccessFull.iloc[tessEnvPredArea.tID.to_list()]


# %%
tessAccess.drop(columns=['buildings']).to_parquet(folder + "accessScore_etc.pq")

# %%
housingTessAoI = gpd.read_parquet(folder + "housingTessAoI.pq")

# %%
housingTessAoI.head(1)

# %%
tessAccess = gpd.read_parquet(folder + "accessScore_etc.pq")
housingTessAoI = gpd.read_parquet(folder + "housingTessAoI.pq")
socialTessClustering = tessAccess.drop(columns="amenityDiversity").merge(housingTessAoI[["tID","HOUSING20","POP20", "housing_score"]], on="tID")

# %%
socialTessClustering.head(1)

# %% [markdown]
# ##### Interesting but out of scope data

# %%
ndvi = dd.read_parquet(folder + "NDVI/*.pq").compute()

# %%
necessaryBuildings = dd.read_parquet(folder + "NecessaryBuildings/*.pq").compute()

# %%
necessaryBuildings = tess.merge(necessaryBuildings, on="tID")
necessaryBuildingsAoI = necessaryBuildings.iloc[tessEnvPredArea.tID.to_list()]

# %%
necessaryBuildingsAoI.to_parquet(folder + "necessaryBuildingsAoI.pq")

# %%
ndvi = ndvi.set_index('tID').sort_index().fillna(0)
ndviAoI=ndvi.iloc[tessEnvPredArea.tID.to_list()]


# %% [markdown]
# We have some categorical values, so we have to encode them as boolean dummies.

# %%
function_data = workplaceAoI.merge(populationAoI, on='tID')#
function_data = function_data.merge(accessibilityAoI, on='tID')#
function_data = function_data.merge(necessaryBuildingsAoI, on='tID').set_index('tID')
function_data = function_data.merge(ndviAoI, on='tID')#.set_index('tID')#


# %% [markdown]
# ##### Clustering of Social Sustainability data

# %%
socialTessClustering = socialTessClustering.set_index('tID')

# %%
socialTessClustering.head(1)

# %%
socialData = socialTessClustering.drop(columns=["tessellation", "housing_score"])

# %%
socialData2 = socialData.drop(columns = ["HOUSING20","POP20"])

# %%
socialTessClustering.to_parquet(folder + "socialTessData.pq")

# %% [markdown]
# Continuous values need to be standardized.

# %%
# scaler = preprocessing.StandardScaler()
# social_data = scaler.fit_transform(socialData)
# social_data[np.isnan(social_data)] = 0

scaler = preprocessing.StandardScaler()
social_data2 = scaler.fit_transform(socialData2)
social_data2[np.isnan(social_data2)] = 0

# %% [markdown]
# ##### Function Data (old)

# %%
scaler = preprocessing.StandardScaler()
f_data = scaler.fit_transform(function_data)

# %% [markdown]
# We can all merge back together.

# %%
f_data[np.isnan(f_data)] = 0

# %% [markdown]
# ##### Social Sustainability Clustering

# %%
cg_social = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_social.fit(social_data)


ax1 = cg_social.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Social Sustainability Silhouette score")

# %%
ax1 = cg_social.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Social Sustainability Silhouette score")
ax1.axvline(6, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(10, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(15, color=c[4], ymin=.02, ymax=.98)


# %% [markdown]
# From the above Silhouette score it is clear that the differentiating factors within the study area are dramatic enough for their to be decently different clusters capable of being made. With peaks at 6, 10, and 15 clusters. 

# %%
ax = cg_social.plot(figsize=(20, 20), linewidth=0.5, cluster_style={"edgecolor": "r", "alpha":.6}, size=1,
       line_style={"alpha":.5})
# ax.set_ylim(-3, 3)

# %% [markdown]
# From the above clustergram, the information seems to be a bit different, there it seems that it is easier to differentiate at 4 clusters and 8 clusters. Thus there are 5 clusters to test, 4, 6, 8, 10, and 15 clusters. 

# %% [markdown]
# Clustergram should be saved to a file.

# %%
with open(folder + 'clustergram_socialFinal.pickle','wb') as f:
    pickle.dump(cg_social, f)

# %%
cg_social = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_social.fit(social_data2)


ax1 = cg_social.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Social Sustainability Silhouette score")

# %%
ax1 = cg_social.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Social Sustainability Silhouette score")
ax1.set_xlim(0, 25)
c = ugg.COLORS
# ax1.axvline(4, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(5, color=c[4], ymin=.02, ymax=.98)
# ax1.axvline(11, color=c[4], ymin=.02, ymax=.98)
# ax1.axvline(13, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# For the second set of clusters for social sustainability, without population data directly included, the cluster peaks occur at 4, 7, 11, and 13 clusters.

# %%
ax = cg_social2.plot(figsize=(20, 20), linewidth=0.5, cluster_style={"edgecolor": "r", "alpha":.6}, size=1,
       line_style={"alpha":.5})
# ax.set_ylim(-3, 3)

# %% [markdown]
# Only other clear cluster of interest is at 6 clusters. Thus the clusters of interest for the second set of social sustainability data is 4, 6, 11, and 13 clusters

# %%
km_social1_4 = KMeans(n_clusters=4, n_init=100).fit(social_data)
km_social1_6 = KMeans(n_clusters=6, n_init=100).fit(social_data)
km_social1_8 = KMeans(n_clusters=8, n_init=100).fit(social_data)
km_social1_10 = KMeans(n_clusters=10, n_init=100).fit(social_data)
km_social1_15 = KMeans(n_clusters=15, n_init=100).fit(social_data)

km_social2_4 = KMeans(n_clusters=4, n_init=100).fit(social_data2)
km_social2_6 = KMeans(n_clusters=6, n_init=100).fit(social_data2)
km_social2_11 = KMeans(n_clusters=11, n_init=100).fit(social_data2)
km_social2_13 = KMeans(n_clusters=13, n_init=100).fit(social_data2)


# %%
km_social = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(social_data2)


# %%
tessSpatialData.head(1)

# %%
tessSpatialData["clusters_social"] = km_social.labels_

# %%
km_social2_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(social_data2)
km_social2_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(social_data2)


# %%
tessSocialData = tessSpatialData.drop(columns=["clusters_form_4", "clusters_form_6", "clusters_form_7", "clusters_form_11", 
                                                "clusters_form_22"]).copy()

# %%
tessSocialData["cluster_social2_7"] = km_social2_7.labels_
tessSocialData["cluster_social2_11"] = km_social2_11.labels_

# %%
tessSocialData["cluster_social1_4"] = km_social1_4.labels_
tessSocialData["cluster_social1_6"] = km_social1_6.labels_
tessSocialData["cluster_social1_8"] = km_social1_8.labels_
tessSocialData["cluster_social1_10"] = km_social1_10.labels_
tessSocialData["cluster_social1_15"] = km_social1_15.labels_

tessSocialData["cluster_social2_4"] = km_social2_4.labels_
tessSocialData["cluster_social2_6"] = km_social2_6.labels_
tessSocialData["cluster_social2_11"] = km_social2_11.labels_
tessSocialData["cluster_social2_13"] = km_social2_13.labels_

# %%
ax = tessSpatialData.plot('clusters_social', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability Clusters")
ax.set_axis_off()
ax = tessSpatialData.set_geometry('buildings').plot('clusters_social', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability Clusters Buildings ")
ax.set_axis_off()

# %%
ax = tessSocialData.plot('cluster_social1_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 1) Clusters ETCs (4 clusters)")
ax.set_axis_off()
ax = tessSocialData.plot('cluster_social1_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 1) Clusters ETCs (6 clusters)")
ax.set_axis_off()
ax = tessSocialData.plot('cluster_social1_8', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 1) Clusters ETCs (8 clusters)")
ax.set_axis_off()
ax = tessSocialData.plot('cluster_social1_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 1) Clusters ETCs (10 clusters)")
ax.set_axis_off()
ax = tessSocialData.plot('cluster_social1_15', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 1) Clusters ETCs (15 clusters)")
ax.set_axis_off()


# %%
ax = tessSocialData.plot('cluster_social2_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 2) Clusters ETCs (4 clusters)")
ax.set_axis_off()
ax = tessSocialData.plot('cluster_social2_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 2) Clusters ETCs (6 clusters)")
ax.set_axis_off()
ax = tessSocialData.plot('cluster_social2_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 2) Clusters ETCs (11 clusters)")
ax.set_axis_off()
ax = tessSocialData.plot('cluster_social2_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 2) Clusters ETCs (13 clusters)")
ax.set_axis_off()

# %%
ax = tessSocialData.plot('cluster_social2_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 2) Clusters ETCs (7 clusters)")
ax.set_axis_off()
ax = tessSocialData.plot('cluster_social2_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 2) Clusters ETCs (11 clusters)")
ax.set_axis_off()

# %%
ax = tessSocialData.set_geometry('buildings').plot('cluster_social1_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 1) Clusters Buildings (4 clusters)")
ax.set_axis_off()
ax = tessSocialData.set_geometry('buildings').plot('cluster_social1_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 1) Clusters Buildings (6 clusters)")
ax.set_axis_off()
ax = tessSocialData.set_geometry('buildings').plot('cluster_social1_8', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 1) Clusters Buildings (8 clusters)")
ax.set_axis_off()
ax = tessSocialData.set_geometry('buildings').plot('cluster_social1_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 1) Clusters Buildings (10 clusters)")
ax.set_axis_off()
ax = tessSocialData.set_geometry('buildings').plot('cluster_social1_15', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 1) Clusters Buildings (15 clusters)")
ax.set_axis_off()


# %%
ax = tessSocialData.set_geometry('buildings').plot('cluster_social2_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 2) Clusters Buildings (4 clusters)")
ax.set_axis_off()
ax = tessSocialData.set_geometry('buildings').plot('cluster_social2_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 2) Clusters Buildings (6 clusters)")
ax.set_axis_off()
ax = tessSocialData.set_geometry('buildings').plot('cluster_social2_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 2) Clusters Buildings (11 clusters)")
ax.set_axis_off()
ax = tessSocialData.set_geometry('buildings').plot('cluster_social2_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Social Sustainability (set 2) Clusters Buildings (13 clusters)")
ax.set_axis_off()

# %%
tessSocialData.to_parquet(folder + "finalSocialTessClustering.pq")

# %% [markdown]
# ### 3.3. Environmental Prediction Clustering
# 
# Clustering based on environmental prediction. Will try to cluster both based on grid cells (so all SUEWS outputs), and on just scoring with tesselation

# %%
tessSocialData.head(1)

# %%
tessEP = tessSocialData.drop(columns=["cluster_social1_4","cluster_social1_6","cluster_social1_8",
                                      "cluster_social1_10","cluster_social1_15","cluster_social2_4",
                                      "cluster_social2_6","cluster_social2_11","cluster_social2_13"]).copy()
tessEP.head(1)

# %%
envScoreGrid = envPredScore.copy().drop(columns=["left", "top", "right", "bottom", "POP_DENSE"])
# envScoreGrid.head(1)
envScore = envScoreGrid.copy().drop(columns=["QH", "QE", "QS", "Tsurf", "Evap", "RO", "TotCh", "Drainage", "SMD"])
# envScore.head(1)

# %%
envQualList = ["QH", "QE", "QS", "Tsurf", "Evap", "RO", "TotCh", "Drainage", "SMD", "heat_score", "water_score"]

# %%
envQual = area_interpolate(envScoreGrid, tessEP, intensive_variables=envQualList, n_jobs=1)


# %%
envQual.to_parquet(folder + "envPredTess.pq")

# %%
envQual = gpd.read_parquet(folder + "envPredTess.pq")

# %%
envQualData = envQual.drop(columns=["geometry"])
scaler = preprocessing.StandardScaler()
envQualData = scaler.fit_transform(envQualData)
envQualData[np.isnan(envQualData)] = 0

# %%
cg_envPred = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_envPred.fit(envQualData)

ax1 = cg_envPred.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Environmental Prediction Silhouette score")

# %%
ax1 = cg_envPred.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Environmental Prediction Silhouette score")
ax1.set_xlim([0, 25])
ax1.axvline(4, color=c[4], ymin=.02, ymax=.98)
# ax1.axvline(7, color=c[4], ymin=.02, ymax=.98)
# ax1.axvline(11, color=c[4], ymin=.02, ymax=.98)
# ax1.axvline(13, color=c[4], ymin=.02, ymax=.98)
# ax1.axvline(30, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# From the silhouette score of the environmental prediction, it can be seen that there is reasonable clustering at 4 clusters, moderately good at 7, and local peaks at 11, 13, and 30 clusters.

# %%
ax = cg_envPred.plot(figsize=(20, 20), linewidth=0.5, cluster_style={"edgecolor": "r", "alpha":.6}, size=1,
       line_style={"alpha":.5})

# %% [markdown]
# From this clustergram, it can be seen that there is some interesting clustering at 6 and 10 clusters. Thus the clusters that will be investigated are 4, 6, 7, 10, 13, and 30.

# %%
with open(folder + 'clustergram_envPred.pickle','wb') as f:
    pickle.dump(cg_envPred, f)

# %%
km_EP_1 = KMeans(n_clusters=4, n_init=100).fit(envQualData)
km_EP_2 = KMeans(n_clusters=6, n_init=100).fit(envQualData)
km_EP_3 = KMeans(n_clusters=7, n_init=100).fit(envQualData)
km_EP_4 = KMeans(n_clusters=10, n_init=100).fit(envQualData)
km_EP_5 = KMeans(n_clusters=13, n_init=100).fit(envQualData)
km_EP_6 = KMeans(n_clusters=30, n_init=100).fit(envQualData)

# %%
km_EP_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(envQualData)
km_EP_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(envQualData)


# %%
km_EP= MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(envQualData)


# %%
tessSpatialData["clusters_env"] = km_EP.labels_

# %%
tessSocialData.head(1)

# %%
tessEnvPredData = tessSocialData[["tID","buildings","tessellation"]].copy()

# %%
tessEnvPredData['clusters_envPred_7'] = km_EP_7.labels_
tessEnvPredData['clusters_envPred_11'] = km_EP_11.labels_

# %%
tessEnvPredData['clusters_envPred_4'] = km_EP_1.labels_
tessEnvPredData['clusters_envPred_6'] = km_EP_2.labels_
tessEnvPredData['clusters_envPred_7'] = km_EP_3.labels_
tessEnvPredData['clusters_envPred_10'] = km_EP_4.labels_
tessEnvPredData['clusters_envPred_13'] = km_EP_5.labels_
tessEnvPredData['clusters_envPred_30'] = km_EP_6.labels_

# %%
ax = tessSpatialData.plot('clusters_env', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters")

ax = tessSpatialData.set_geometry('buildings').plot('clusters_env', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters Buildings")

# %%
ax = tessEnvPredData.plot('clusters_envPred_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters ETCs (7)")

ax = tessEnvPredData.plot('clusters_envPred_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters ETCs (10)")


# %%
ax = tessEnvPredData.plot('clusters_envPred_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters ETCs (4)")

ax = tessEnvPredData.plot('clusters_envPred_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters ETCs (6)")

ax = tessEnvPredData.plot('clusters_envPred_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters ETCs (7)")

ax = tessEnvPredData.plot('clusters_envPred_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters ETCs (10)")

ax = tessEnvPredData.plot('clusters_envPred_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters ETCs (13)")

ax = tessEnvPredData.plot('clusters_envPred_30', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters ETCs (30)")

ax = tessEnvPredData.set_geometry('buildings').plot('clusters_envPred_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters Buildings (4)")

ax = tessEnvPredData.set_geometry('buildings').plot('clusters_envPred_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters Buildings (6)")

ax = tessEnvPredData.set_geometry('buildings').plot('clusters_envPred_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters Buildings (7)")

ax = tessEnvPredData.set_geometry('buildings').plot('clusters_envPred_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters Buildings (10)")

ax = tessEnvPredData.set_geometry('buildings').plot('clusters_envPred_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters Buildings (13)")

ax = tessEnvPredData.set_geometry('buildings').plot('clusters_envPred_30', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters Buildings (30)")

# %% [markdown]
# From the above plots, it can be seen that for environmental predictions, the most meaningful clustering occurs at 7 clusters and 10 clusters.

# %%
tessEnvPredData.to_parquet(folder + "tessEnvPredClusterFinal.pq")

# %% [markdown]
# ##### Older EP clustering (no longer useful)

# %%
envScoresList = ["heat_score", "water_score"]

# %%
envScore = area_interpolate(envScore, tessEP, intensive_variables=envScoresList, n_jobs=1)


# %%
envScoreGridData = envScoreGrid.drop(columns=["geometry"])

# %%
scaler = preprocessing.StandardScaler()
envScoreGridData = scaler.fit_transform(envScoreGridData)
envScoreGridData[np.isnan(envScoreGridData)] = 0

# %%
envScoreData = envScore.drop(columns=["geometry"])

# %%
scaler = preprocessing.StandardScaler()
envScoreData = scaler.fit_transform(envScoreData)
envScoreData[np.isnan(envScoreData)] = 0

# %%
cg_envPredGrid = Clustergram(range(1, 30), backend='sklearn', n_init=30)
cg_envPredGrid.fit(envScoreGridData)

# %%
ax = cg_envPredGrid.plot(figsize=(20, 20), linewidth=0.5, cluster_style={"edgecolor": "r", "alpha":.6}, size=1,
       line_style={"alpha":.5})

# %% [markdown]
# 9 and 11 clusters seem to be right

# %%
with open(folder + 'clustergram_envGrid.pickle','wb') as f:
    pickle.dump(cg_envPredGrid, f)

# %%
km_function_EP_Grid_9 = KMeans(n_clusters=9, n_init=100).fit(envScoreGridData)
km_function_EP_Grid_11 = KMeans(n_clusters=11, n_init=100).fit(envScoreGridData)

# %%
envScoreGrid['clusters_envPred_9'] = km_function_EP_Grid_9.labels_
envScoreGrid['clusters_envPred_11'] = km_function_EP_Grid_11.labels_

# %%
ax = envScoreGrid.plot('clusters_envPred_9', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters Grid (9)")
ax = envScoreGrid.plot('clusters_envPred_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters Grid (11)")

# %%
cg_envPred = Clustergram(range(1, 30), backend='sklearn', n_init=30)
cg_envPred.fit(envScoreData)

# %%
ax = cg_envPred.plot(figsize=(20, 20), linewidth=0.5, cluster_style={"edgecolor": "r", "alpha":.6}, size=1,
       line_style={"alpha":.5})

# %% [markdown]
# again 9 and 11 clusters

# %%
with open(folder + 'clustergram_env.pickle','wb') as f:
    pickle.dump(cg_envPred, f)

# %%
km_function_EP_9 = KMeans(n_clusters=9, n_init=100).fit(envScoreData)
km_function_EP_11 = KMeans(n_clusters=11, n_init=100).fit(envScoreData)

# %%
tessEnvPredArea['clusters_envPred_9'] = km_function_EP_9.labels_
tessEnvPredArea['clusters_envPred_11'] = km_function_EP_11.labels_

# %%
ax = tessEnvPredArea.plot('clusters_envPred_9', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters ETCs (9)")
ax = tessEnvPredArea.set_geometry('buildings').plot('clusters_envPred_9', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters Buildings (9)")
ax = tessEnvPredArea.plot('clusters_envPred_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters ETCs (11)")
ax = tessEnvPredArea.set_geometry('buildings').plot('clusters_envPred_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction Clusters Buildings (11)")

# %%
tessEnvPredArea.to_parquet(folder + "tess_clusters_spatial_social_environmental_nonCombined.pq")

# %% [markdown]
# It is clear that interpolation from grid spaces do not represent

# %% [markdown]
# ### 3.4. Spatial and Social Signatures
# 
# Final clustering combines both form and function into a singular classification.
# 
# First we stack both preprocessed arrays together and then run clustergram.

# %%
#combined convolution data
# formAoI_gdf1 = gpd.read_parquet(folder + "convolutions_AoI_usedCharsFullFinal.pq")
formAoI_gdf = gpd.read_parquet(folder + "convolutions_AoI_usedCharsFinal.pq")

# %%
#combined chunk data
chunks_gdf = gpd.read_parquet(folder + "combChunks.pq")

# %%
# scaler = preprocessing.StandardScaler()
# data1 = scaler.fit_transform(formAoI_gdf1.drop(columns=["geometry"]))
# data1[np.isnan(data1)] = 0
scaler = preprocessing.StandardScaler()
spatial_data = scaler.fit_transform(formAoI_gdf.drop(columns=["geometry"]))
spatial_data[np.isnan(spatial_data)] = 0

# %%
social_data_gdf = gpd.read_parquet(folder + "socialTessData.pq")

# %%
social_data_gdf.head(1)


# %%
scaler = preprocessing.StandardScaler()
social_data1 = scaler.fit_transform(social_data_gdf.drop(columns=["tessellation", 
                                                                  "housing_score"]))
social_data1[np.isnan(social_data1)] = 0

scaler = preprocessing.StandardScaler()
social_data2 = scaler.fit_transform(social_data_gdf.drop(columns=["tessellation",
                                                                  "HOUSING20",
                                                                  "POP20", 
                                                                  "housing_score"]))
social_data2[np.isnan(social_data2)] = 0

# %%
scaler = preprocessing.StandardScaler()
socialScore_data = scaler.fit_transform(social_data_gdf.drop(columns=["tessellation", "clsCnt", "btwCnt", "strCnt",
                                                                  "HOUSING20",
                                                                  "POP20", 
                                                                  "housing_score"]))
socialScore_data[np.isnan(socialScore_data)] = 0

# %%
social1Spatial_data = np.hstack([spatial_data,social_data1])
social2Spatial_data = np.hstack([spatial_data,social_data2])
# socialSpatial_data2 = np.hstack([data2,social_data])

# %%
cg_s1s = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_s1s.fit(social1Spatial_data)
ax1 = cg_s1s.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Silhouette score")

# %%
ax1 = cg_s1s.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability (set 1) Silhouette score")
ax1.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(12, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(15, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# CLusters at 3, 12, and 15

# %%
ax = cg_s1s.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgram_s1s.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# Clusters 5 and 8 from here. Meaning for set 1 of social sustainability metrics, clusters of interest are 3, 5, 8, 12, and 15.

# %%
cg_s2s = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_s2s.fit(social2Spatial_data)
ax2 = cg_s2s.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form + Social Sustainability (set 2) Silhouette score")

# %%
ax2 = cg_s2s.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form + Social Sustainability (set 2) Silhouette score")
ax2.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(9, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(11, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(20, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# CLusters of 3, 9, 11, and 20

# %%
ax = cg_s2s.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgram_s2s.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# Clusters 5 and 7 are of interest. Meaning clusters of interest are: 3, 5, 7, 9, 11, and 20.

# %%
# 3, 5, 8, 12, and 15.
km_ss1_3 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(social1Spatial_data)
km_ss1_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(social1Spatial_data)
km_ss1_8 = MiniBatchKMeans(n_clusters=8, n_init=1000, random_state=12345, batch_size=100000).fit(social1Spatial_data)
km_ss1_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(social1Spatial_data)
km_ss1_15 = MiniBatchKMeans(n_clusters=15, n_init=1000, random_state=12345, batch_size=100000).fit(social1Spatial_data)

km_ss2_3 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(social2Spatial_data)
km_ss2_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(social2Spatial_data)
km_ss2_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(social2Spatial_data)
km_ss2_9 = MiniBatchKMeans(n_clusters=9, n_init=1000, random_state=12345, batch_size=100000).fit(social2Spatial_data)
km_ss2_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(social2Spatial_data)
km_ss2_20 = MiniBatchKMeans(n_clusters=20, n_init=1000, random_state=12345, batch_size=100000).fit(social2Spatial_data)


# %%
tessSocialData.head(1)

# %%
tessFormAndSocialData = tessSocialData.drop(columns=["cluster_social1_4", "cluster_social1_6", "cluster_social1_8", 
                                                      "cluster_social1_10","cluster_social1_15", "cluster_social2_4",
                                                      "cluster_social2_6","cluster_social2_11","cluster_social2_13"]).copy()

# %%
tessFormAndSocialData["cluster_form_and_social1_3"] = km_ss1_3.labels_
tessFormAndSocialData["cluster_form_and_social1_5"] = km_ss1_5.labels_
tessFormAndSocialData["cluster_form_and_social1_8"] = km_ss1_8.labels_
tessFormAndSocialData["cluster_form_and_social1_12"] = km_ss1_12.labels_
tessFormAndSocialData["cluster_form_and_social1_15"] = km_ss1_15.labels_

tessFormAndSocialData["cluster_form_and_social2_3"] = km_ss2_3.labels_
tessFormAndSocialData["cluster_form_and_social2_5"] = km_ss2_5.labels_
tessFormAndSocialData["cluster_form_and_social2_7"] = km_ss2_7.labels_
tessFormAndSocialData["cluster_form_and_social2_9"] = km_ss2_9.labels_
tessFormAndSocialData["cluster_form_and_social2_11"] = km_ss2_11.labels_
tessFormAndSocialData["cluster_form_and_social2_20"] = km_ss2_20.labels_

# %%
ax = tessFormAndSocialData.plot('cluster_form_and_social1_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 1) Clusters ETCs (3)")

ax = tessFormAndSocialData.plot('cluster_form_and_social1_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 1) Clusters ETCs (5)")

ax = tessFormAndSocialData.plot('cluster_form_and_social1_8', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 1) Clusters ETCs (8)")

ax = tessFormAndSocialData.plot('cluster_form_and_social1_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 1) Clusters ETCs (12)")

ax = tessFormAndSocialData.plot('cluster_form_and_social1_15', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 1) Clusters ETCs (15)")


ax = tessFormAndSocialData.set_geometry('buildings').plot('cluster_form_and_social1_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 1) Clusters Buildings (3)")

ax = tessFormAndSocialData.set_geometry('buildings').plot('cluster_form_and_social1_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 1) Clusters Buildings (5)")

ax = tessFormAndSocialData.set_geometry('buildings').plot('cluster_form_and_social1_8', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 1) Clusters Buildings (8)")

ax = tessFormAndSocialData.set_geometry('buildings').plot('cluster_form_and_social1_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 1) Clusters Buildings (12)")

ax = tessFormAndSocialData.set_geometry('buildings').plot('cluster_form_and_social1_15', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 1) Clusters Buildings (15)")

# %%
ax = tessFormAndSocialData.plot('cluster_form_and_social2_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 2) Clusters ETCs (3)")

ax = tessFormAndSocialData.plot('cluster_form_and_social2_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 2) Clusters ETCs (5)")

ax = tessFormAndSocialData.plot('cluster_form_and_social2_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 2) Clusters ETCs (7)")

ax = tessFormAndSocialData.plot('cluster_form_and_social2_9', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 2) Clusters ETCs (9)")

ax = tessFormAndSocialData.plot('cluster_form_and_social2_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 2) Clusters ETCs (11)")

ax = tessFormAndSocialData.plot('cluster_form_and_social2_20', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 2) Clusters ETCs (20)")

ax = tessFormAndSocialData.set_geometry('buildings').plot('cluster_form_and_social2_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 2) Clusters Buildings (3)")

ax = tessFormAndSocialData.set_geometry('buildings').plot('cluster_form_and_social2_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 2) Clusters Buildings (5)")

ax = tessFormAndSocialData.set_geometry('buildings').plot('cluster_form_and_social2_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 2) Clusters Buildings (7)")

ax = tessFormAndSocialData.set_geometry('buildings').plot('cluster_form_and_social2_9', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 2) Clusters Buildings (9)")

ax = tessFormAndSocialData.set_geometry('buildings').plot('cluster_form_and_social2_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 2) Clusters Buildings (11)")

ax = tessFormAndSocialData.set_geometry('buildings').plot('cluster_form_and_social2_20', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability (Set 2) Clusters Buildings (20)")

# %%
tessFormAndSocialData.to_parquet(folder + "spatialSocialTessFinal.pq")

# %% [markdown]
# ##### Older social spatial clustering

# %%
cg_ss1 = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_ss1.fit(socialSpatial_data1)
ax1 = cg_ss1.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (full) + Social Sustainability Silhouette score")

# %%
# ax1.axvline(2, color=c[4], ymin=.02, ymax=.98)
# ax1.axvline(10, color=c[4], ymin=.02, ymax=.98)
# ax1.axvline(14, color=c[4], ymin=.02, ymax=.98)
# ax1.axvline(19, color=c[4], ymin=.02, ymax=.98)
cg_ss2 = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_ss2.fit(socialSpatial_data2)
ax2 = cg_ss2.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (selected) + Social Sustainability Silhouette score")


# %% [markdown]
# Clearly, it is of interest to only use the selected spatial indicators, since when all indicators are used it is clear that there are less than adequate ability to distinguish between clusters. 

# %%

ax2 = cg_ss2.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (selected) + Social Sustainability Silhouette score")
ax2.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(4, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(10, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(13, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# From this it can be seen that there are clusters of interest at 3, 4, 10, and 13.

# %%
c = ugg.COLORS
ax = cg_ss2.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgram_ss.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# The clusters of interest are at 6, 8, and 13 clusters. Thus the clusters will be done with 3, 4, 6, 8, 10, and 13 clusters.

# %%
km_ss_1 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(socialSpatial_data2)
km_ss_2 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(socialSpatial_data2)
km_ss_3 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(socialSpatial_data2)
km_ss_4 = MiniBatchKMeans(n_clusters=8, n_init=1000, random_state=12345, batch_size=100000).fit(socialSpatial_data2)
km_ss_5 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(socialSpatial_data2)
km_ss_6 = MiniBatchKMeans(n_clusters=13, n_init=1000, random_state=12345, batch_size=100000).fit(socialSpatial_data2)

# %%
tessFormAndSocialData = tessEnvPredData.drop(columns=["clusters_envPred_4", "clusters_envPred_6", "clusters_envPred_7", 
                                                      "clusters_envPred_10","clusters_envPred_13", "clusters_envPred_30"]).copy()

# %%
tessFormAndSocialData["cluster_form_and_social_3"] = km_ss_1.labels_
tessFormAndSocialData["cluster_form_and_social_4"] = km_ss_2.labels_
tessFormAndSocialData["cluster_form_and_social_6"] = km_ss_3.labels_
tessFormAndSocialData["cluster_form_and_social_8"] = km_ss_4.labels_
tessFormAndSocialData["cluster_form_and_social_10"] = km_ss_5.labels_
tessFormAndSocialData["cluster_form_and_social_13"] = km_ss_6.labels_

# %%
ax = tessFormAndSocialData.plot('cluster_form_and_social_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability Clusters ETCs (3)")

ax = tessFormAndSocialData.plot('cluster_form_and_social_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability Clusters ETCs (4)")

ax = tessFormAndSocialData.plot('cluster_form_and_social_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability Clusters ETCs (6)")

ax = tessFormAndSocialData.plot('cluster_form_and_social_8', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability Clusters ETCs (8)")

ax = tessFormAndSocialData.plot('cluster_form_and_social_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability Clusters ETCs (10)")

ax = tessFormAndSocialData.plot('cluster_form_and_social_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability Clusters ETCs (13)")

ax = tessFormAndSocialData.set_geometry('buildings').plot('cluster_form_and_social_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability Clusters Buildings (3)")

ax = tessFormAndSocialData.set_geometry('buildings').plot('cluster_form_and_social_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability Clusters Buildings (4)")

ax = tessFormAndSocialData.set_geometry('buildings').plot('cluster_form_and_social_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability Clusters Buildings (6)")

ax = tessFormAndSocialData.set_geometry('buildings').plot('cluster_form_and_social_8', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability Clusters Buildings (8)")

ax = tessFormAndSocialData.set_geometry('buildings').plot('cluster_form_and_social_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability Clusters Buildings (10)")

ax = tessFormAndSocialData.set_geometry('buildings').plot('cluster_form_and_social_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Social Sustainability Clusters Buildings (13)")

# %% [markdown]
# From the plots abovve, it seems that the most representative of the data, based on the experiential knowledge of the author, are 7, 8, and 10 clusters. 

# %%
tessFormAndSocialData.to_parquet(folder + "spatialSocialTess_final.pq")

# %% [markdown]
# ##### Old Data

# %%
c = ugg.COLORS
ax = cg_ss2.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgram_ss1.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# 12, 13, 14 clusters

# %%
cg_labels = cg_ss1.labels
cg_labels.columns = cg_labels.columns.astype(str)
cg_labels.to_parquet(folder + 'clustergram_ss1_labels.pq')
cg_labels = cg_ss2.labels
cg_labels.columns = cg_labels.columns.astype(str)
cg_labels.to_parquet(folder + 'clustergram_ss2_labels.pq')

# %%
km_13 = MiniBatchKMeans(n_clusters=13, n_init=1000, random_state=12345, batch_size=100000).fit(socialSpatial_data1)


# %%
km_13 = MiniBatchKMeans(n_clusters=13, n_init=1000, random_state=12345, batch_size=100000).fit(socialSpatial_data1)
km_19 = MiniBatchKMeans(n_clusters=19, n_init=1000, random_state=12345, batch_size=100000).fit(socialSpatial_data1)
km_21 = MiniBatchKMeans(n_clusters=21, n_init=1000, random_state=12345, batch_size=100000).fit(socialSpatial_data1)

# %%
km2_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(socialSpatial_data2)
km2_13 = MiniBatchKMeans(n_clusters=13, n_init=1000, random_state=12345, batch_size=100000).fit(socialSpatial_data2)
km2_14 = MiniBatchKMeans(n_clusters=14, n_init=1000, random_state=12345, batch_size=100000).fit(socialSpatial_data2)

# %%
tessEnvPredArea['clusters_social_spatial_13'] = km_13.labels_
tessEnvPredArea['clusters_social_spatial_19'] = km_19.labels_
tessEnvPredArea['clusters_social_spatial_21'] = km_21.labels_
tessEnvPredArea.to_parquet(folder + "socialSpatialCluster1_tess.pq")

# %%
tessEnvPredArea['clusters_social_spatial2_12'] = km2_12.labels_
tessEnvPredArea['clusters_social_spatial2_13'] = km2_13.labels_
tessEnvPredArea['clusters_social_spatial2_14'] = km2_14.labels_
tessEnvPredArea.drop(columns=["clusters_social_spatial_13","clusters_social_spatial_19","clusters_social_spatial_21"]).to_parquet(folder + "socialSpatialCluster2_tess.pq")
tessEnvPredArea.to_parquet(folder + "socialSpatialCluster_tess.pq")

# %%
ax = tessEnvPredArea.plot('clusters_social_spatial_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial set 1 + Social Form Clusters (13 clusters)")
ax.set_axis_off()
ax = tessEnvPredArea.plot('clusters_social_spatial_19', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial set 1 + Social Form Clusters (19 clusters)")
ax.set_axis_off()
ax = tessEnvPredArea.plot('clusters_social_spatial_21', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial set 1 + Social Form Clusters (21 clusters)")
ax.set_axis_off()

# %%
ax = tessEnvPredArea.plot('clusters_social_spatial2_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial set 2 + Social Form Clusters (12 clusters)")
ax.set_axis_off()
ax = tessEnvPredArea.plot('clusters_social_spatial2_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial set 2 + Social Form Clusters (13 clusters)")
ax.set_axis_off()
ax = tessEnvPredArea.plot('clusters_social_spatial2_14', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial set 2 + Social Form Clusters (14 clusters)")
ax.set_axis_off()

# %% [markdown]
# ### 3.5. Spatial and Environmental Signatures
# 
# Clustering based on spatial and environmental signatures. Using grid based clustering to ensure that the environmental clustering is reflective of the output of UMEP. 

# %%
#combined convolution data
formAoI_gdf = gpd.read_parquet(folder + "convolutions_AoI_usedCharsFinal.pq")
# formAoI_gdf1 = gpd.read_parquet(folder + "convolutions_AoI_usedCharsFullFinal.pq")
# formAoI_gdf2 = gpd.read_parquet(folder + "convolutions_AoI_usedCharsFinal.pq")

# %%
scaler = preprocessing.StandardScaler()
dataSpatial = scaler.fit_transform(formAoI_gdf.drop(columns=["geometry"]))
dataSpatial[np.isnan(dataSpatial)] = 0
# scaler = preprocessing.StandardScaler()
# data2 = scaler.fit_transform(formAoI_gdf2.drop(columns=["geometry"]))
# data2[np.isnan(data2)] = 0

# %%
envPredTess = gpd.read_parquet(folder + "envPredTess.pq")

# %%
scaler = preprocessing.StandardScaler()
envPred_data = scaler.fit_transform(envPredTess.drop(columns=["geometry"]))
envPred_data[np.isnan(envPred_data)] = 0

scaler = preprocessing.StandardScaler()
envPredScores_data = scaler.fit_transform(envPredTess.drop(columns=["QH","QE","QS",
                                                                    "Tsurf","Evap",
                                                                    "RO","TotCh",
                                                                    "Drainage","SMD",
                                                                    "geometry"]))
envPredScores_data[np.isnan(envPredScores_data)] = 0

scaler = preprocessing.StandardScaler()
envPredChar_data = scaler.fit_transform(envPredTess.drop(columns=["heat_score",
                                                                  "water_score",
                                                                  "geometry"]))
envPredChar_data[np.isnan(envPredChar_data)] = 0

# %%
spatialEnvPred_data = np.hstack([dataSpatial,envPred_data])
spatialEnvPredScores_data = np.hstack([dataSpatial,envPredScores_data])
spatialEnvPredChar_data = np.hstack([dataSpatial,envPredChar_data])

# spatialEnvPred_data1 = np.hstack([data1,envPred_data])
# spatialEnvPred_data2 = np.hstack([data2,envPred_data])

# %%
cg_sEP = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_sEP.fit(spatialEnvPred_data)
ax1 = cg_sEP.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Environmental Prediction Silhouette score")

# %%
ax1 = cg_sEP.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Environmental Prediction Silhouette score")
ax1.axvline(4, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(10, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(14, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# Clusters of interest are 4, 10, and 14

# %%
ax = cg_sEP.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgram_sEP.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# 5 is a cluster of interest here, thus clusters of interest are 4, 5, 10, and 14

# %%
cg_sEPScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_sEPScores.fit(spatialEnvPredScores_data)
ax1 = cg_sEPScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Environmental Prediction Scores Silhouette score")

# %%
ax1 = cg_sEPScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Environmental Prediction Scores Silhouette score")
ax1.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(10, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(22, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# Clusters of interest are 3, 10, and 22

# %%
ax = cg_sEPScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgram_sEPScores.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# 5 and 8 are clusters of interest. Thus clusters of interest for the EP scores and spatial signatures are: 3, 5, 8, 10, and 22

# %%
cg_sEPChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_sEPChars.fit(spatialEnvPredChar_data)
ax1 = cg_sEPChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Environmental Prediction Characteristics Silhouette score")

# %%
ax1 = cg_sEPChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Environmental Prediction Characteristics Silhouette score")
ax1.axvline(5, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(11, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(13, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# The clusters of interest occur at 5, 11, and 13.

# %%
ax = cg_sEPChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgram_sEPChars.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# Clusters of interest occur at 6, thus the clusters of interest are at 5, 6, 11, and 13.

# %%
#4, 5, 10, and 14 
km_sEP_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPred_data)
km_sEP_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPred_data)
km_sEP_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPred_data)
km_sEP_14 = MiniBatchKMeans(n_clusters=14, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPred_data)

# 3, 5, 8, 10, and 22
km_sEPScores_3 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPredScores_data)
km_sEPScores_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPredScores_data)
km_sEPScores_8 = MiniBatchKMeans(n_clusters=8, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPredScores_data)
km_sEPScores_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPredScores_data)
km_sEPScores_22 = MiniBatchKMeans(n_clusters=22, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPredScores_data)

km_sEPChar_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPredChar_data)
km_sEPChar_6 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPredChar_data)
km_sEPChar_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPredChar_data)
km_sEPChar_13 = MiniBatchKMeans(n_clusters=13, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPredChar_data)


# %%
km_sEPScores_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPredScores_data)
km_sEPScores_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPredScores_data)

km_sEPChar_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPredChar_data)
km_sEPChar_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPredChar_data)


# %%
tessFormAndEPData = tessFormAndSocialData.drop(columns=["cluster_form_and_social1_3", "cluster_form_and_social1_5", "cluster_form_and_social1_5", 
                                                      "cluster_form_and_social1_12","cluster_form_and_social1_15", "cluster_form_and_social2_3",
                                                      "cluster_form_and_social2_5","cluster_form_and_social2_7","cluster_form_and_social2_9",
                                                      "cluster_form_and_social2_11","cluster_form_and_social2_20"]).copy()

# %%
tessFormAndEPData["cluster_form_and_EnvPredScores_7"] = km_sEPScores_7.labels_
tessFormAndEPData["cluster_form_and_EnvPredScores_11"] = km_sEPScores_11.labels_

tessFormAndEPData["cluster_form_and_EnvPredChar_7"] = km_sEPChar_7.labels_
tessFormAndEPData["cluster_form_and_EnvPredChar_11"] = km_sEPChar_11.labels_


# %%
tessFormAndEPData["cluster_form_and_EnvPred_4"] = km_sEP_4.labels_
tessFormAndEPData["cluster_form_and_EnvPred_5"] = km_sEP_5.labels_
tessFormAndEPData["cluster_form_and_EnvPred_10"] = km_sEP_10.labels_
tessFormAndEPData["cluster_form_and_EnvPred_14"] = km_sEP_14.labels_

tessFormAndEPData["cluster_form_and_EnvPredScores_3"] = km_sEPScores_3.labels_
tessFormAndEPData["cluster_form_and_EnvPredScores_5"] = km_sEPScores_5.labels_
tessFormAndEPData["cluster_form_and_EnvPredScores_8"] = km_sEPScores_8.labels_
tessFormAndEPData["cluster_form_and_EnvPredScores_10"] = km_sEPScores_10.labels_
tessFormAndEPData["cluster_form_and_EnvPredScores_22"] = km_sEPScores_22.labels_

tessFormAndEPData["cluster_form_and_EnvPredChar_5"] = km_sEPChar_5.labels_
tessFormAndEPData["cluster_form_and_EnvPredChar_6"] = km_sEPChar_6.labels_
tessFormAndEPData["cluster_form_and_EnvPredChar_11"] = km_sEPChar_11.labels_
tessFormAndEPData["cluster_form_and_EnvPredChar_13"] = km_sEPChar_13.labels_

# %%
ax = tessFormAndEPData.plot('cluster_form_and_EnvPred_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Clusters ETCs (4)")

ax = tessFormAndEPData.plot('cluster_form_and_EnvPred_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Clusters ETCs (5)")

ax = tessFormAndEPData.plot('cluster_form_and_EnvPred_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Clusters ETCs (10)")

ax = tessFormAndEPData.plot('cluster_form_and_EnvPred_14', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Clusters ETCs (14)")


ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EnvPred_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Clusters Buildings (4)")

ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EnvPred_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Clusters Buildings (5)")

ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EnvPred_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Clusters Buildings (10)")

ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EnvPred_14', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Clusters Buildings (14)")


# %%
ax = tessFormAndEPData.plot('cluster_form_and_EnvPredScores_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Scores Clusters ETCs (7)")

ax = tessFormAndEPData.plot('cluster_form_and_EnvPredScores_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Scores Clusters ETCs (11)")

ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EnvPredScores_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Scores Clusters Buildings (7)")

ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EnvPredScores_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Scores Clusters Buildings (11)")

# %%
ax = tessFormAndEPData.plot('cluster_form_and_EnvPredScores_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Scores Clusters ETCs (3)")

ax = tessFormAndEPData.plot('cluster_form_and_EnvPredScores_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Scores Clusters ETCs (5)")

ax = tessFormAndEPData.plot('cluster_form_and_EnvPredScores_8', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Scores Clusters ETCs (8)")

ax = tessFormAndEPData.plot('cluster_form_and_EnvPredScores_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Scores Clusters ETCs (10)")

ax = tessFormAndEPData.plot('cluster_form_and_EnvPredScores_22', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Scores Clusters ETCs (22)")


ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EnvPredScores_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Scores Clusters Buildings (3)")

ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EnvPredScores_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Scores Clusters Buildings (5)")

ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EnvPredScores_8', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Scores Clusters Buildings (8)")

ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EnvPredScores_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Scores Clusters Buildings (10)")

ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EnvPredScores_22', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Scores Clusters Buildings (22)")

# %%
ax = tessFormAndEPData.plot('cluster_form_and_EnvPredChar_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Characteristics Clusters ETCs (5)")

ax = tessFormAndEPData.plot('cluster_form_and_EnvPredChar_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Characteristics Clusters ETCs (6)")

ax = tessFormAndEPData.plot('cluster_form_and_EnvPredChar_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Characteristics Clusters ETCs (11)")

ax = tessFormAndEPData.plot('cluster_form_and_EnvPredChar_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Characteristics Clusters ETCs (13)")


ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EnvPredChar_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Characteristics Clusters Buildings (5)")

ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EnvPredChar_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Characteristics Clusters Buildings (6)")

ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EnvPredChar_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Characteristics Clusters Buildings (11)")

ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EnvPredChar_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Characteristics Clusters Buildings (13)")

# %%
tessFormAndEPData.to_parquet(folder + "spatialEnvPredTessFinal.pq")

# %% [markdown]
# ##### Old clustering

# %%
cg_sEP = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_sEP.fit(spatialEnvPred_data1)
ax1 = cg_sEP.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (full) + Environmental Prediction Silhouette score")

# %%
cg_sEP2 = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_sEP2.fit(spatialEnvPred_data2)
ax2 = cg_sEP2.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (selected) + Environmental Prediction Silhouette score")

# %%
ax2 = cg_sEP2.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (selected) + Environmental Prediction Silhouette score")
ax2.axvline(4, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(11, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(14, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# From the silhouette score it can be seen that the clusters of interest are at 4, 11, and 14 clusters. However, all of these clusters have weak differentiation. 

# %%
c = ugg.COLORS
ax = cg_sEP2.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgram_sEP.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# From the clustergram it can be seen that the clusters of interest are at 12 and 17 clusters. Thus the KMeans will be done with 4, 11, 12, 14, and 17 clusters.

# %%
km_sEP_1 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPred_data2)
km_sEP_2 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPred_data2)
km_sEP_3 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPred_data2)
km_sEP_4 = MiniBatchKMeans(n_clusters=14, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPred_data2)
km_sEP_5 = MiniBatchKMeans(n_clusters=17, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvPred_data2)

# %%
tessFormAndEPData = tessFormAndSocialData.drop(columns=["cluster_form_and_social_3", "cluster_form_and_social_4", 
                                                        "cluster_form_and_social_6", "cluster_form_and_social_8",
                                                        "cluster_form_and_social_10", "cluster_form_and_social_13"]).copy()

# %%
tessFormAndEPData["cluster_form_and_EP_4"] = km_sEP_1.labels_
tessFormAndEPData["cluster_form_and_EP_11"] = km_sEP_2.labels_
tessFormAndEPData["cluster_form_and_EP_12"] = km_sEP_3.labels_
tessFormAndEPData["cluster_form_and_EP_14"] = km_sEP_4.labels_
tessFormAndEPData["cluster_form_and_EP_17"] = km_sEP_5.labels_


# %%
ax = tessFormAndEPData.plot('cluster_form_and_EP_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Clusters ETCs (4)")

ax = tessFormAndEPData.plot('cluster_form_and_EP_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Clusters ETCs (11)")

ax = tessFormAndEPData.plot('cluster_form_and_EP_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Clusters ETCs (12)")

ax = tessFormAndEPData.plot('cluster_form_and_EP_14', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Clusters ETCs (14)")

ax = tessFormAndEPData.plot('cluster_form_and_EP_17', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Clusters ETCs (17)")


ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EP_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Clusters Buildings (4)")

ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EP_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Clusters Buildings (11)")

ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EP_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Clusters Buildings (12)")

ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EP_14', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Clusters Buildings (14)")

ax = tessFormAndEPData.set_geometry('buildings').plot('cluster_form_and_EP_17', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial and Environmental Prediction Clusters Buildings (17)")

# %% [markdown]
# From observation, 4 and 11 clusters make the most sense, however this would need to be validated somehow

# %%
tessFormAndEPData.to_parquet(folder + "spatialEnvPredTess_final.pq")

# %% [markdown]
# ##### Grid Calculation - less interesting

# %%
scaler = preprocessing.StandardScaler()
dataGrid1 = scaler.fit_transform(spatialSignaturesTrends1.drop(columns=["geometry", "clusters_form_8", "clusters_form_15", "clusters_form_20"]))
dataGrid1[np.isnan(dataGrid1)] = 0
scaler = preprocessing.StandardScaler()
dataGrid2 = scaler.fit_transform(spatialSignaturesTrends2.drop(columns=["geometry", "clusters_form_8", "clusters_form_13"]))
dataGrid2[np.isnan(dataGrid2)] = 0

# %%
spatialEnvGrid1 = spatialSignaturesTrends1.drop(columns=["clusters_form_8", "clusters_form_15","clusters_form_20"]).merge(envScoreGrid.drop(columns=["clusters_envPred_9","clusters_envPred_11"]), on="geometry")
spatialEnvGrid2 = spatialSignaturesTrends2.drop(columns=["clusters_form_8", "clusters_form_13"]).merge(envScoreGrid.drop(columns=["clusters_envPred_9","clusters_envPred_11"]), on="geometry")

# %%
spatialEnvData1 = np.hstack([dataGrid1, envScoreGridData])
spatialEnvData2 = np.hstack([dataGrid2, envScoreGridData])

# %%
cg_spatialEnv1 = Clustergram(range(1, 30), n_init=30, random_state=12345)
cg_spatialEnv1.fit(spatialEnvData1)
cg_spatialEnv2 = Clustergram(range(1, 30), n_init=30, random_state=12345)
cg_spatialEnv2.fit(spatialEnvData2)

# %%
c = ugg.COLORS
ax = cg_spatialEnv1.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cSpatialEnvgram1.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# Clusters at 9, 13, 17

# %%
cg_labels = cg_spatialEnv1.labels
cg_labels.columns = cg_labels.columns.astype(str)
cg_labels.to_parquet(folder + 'clustergram_spatialEnv1_labels.pq')

# %%
km1_9 = MiniBatchKMeans(n_clusters=9, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvData1)
km1_13 = MiniBatchKMeans(n_clusters=13, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvData1)
km1_17 = MiniBatchKMeans(n_clusters=17, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvData1)

# %%
spatialEnvGrid1['clusters_env_spatial1_13'] = km1_9.labels_
spatialEnvGrid1['clusters_env_spatial1_19'] = km1_13.labels_
spatialEnvGrid1['clusters_env_spatial1_21'] = km1_17.labels_
spatialEnvGrid1.to_parquet(folder + "spatialEnvChar1_cluster_grid.pq")


# %%
ax = spatialEnvGrid1.plot('clusters_env_spatial1_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction and Spatial Signatures set 1 Clusters Grid (13)")
ax = spatialEnvGrid1.plot('clusters_env_spatial1_19', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction and Spatial Signatures set 1 Clusters Grid (19)")
ax = spatialEnvGrid1.plot('clusters_env_spatial1_21', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction and Spatial Signatures set 1 Clusters Grid (21)")

# %%
c = ugg.COLORS
ax = cg_spatialEnv2.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cSpatialEnvgram2.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# Clusters at 8, 16, 22

# %%
cg_labels = cg_spatialEnv2.labels
cg_labels.columns = cg_labels.columns.astype(str)
cg_labels.to_parquet(folder + 'clustergram_spatialEnv2_labels.pq')

# %%
km2_8 = MiniBatchKMeans(n_clusters=8, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvData2)
km2_16 = MiniBatchKMeans(n_clusters=16, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvData2)
km2_22 = MiniBatchKMeans(n_clusters=22, n_init=1000, random_state=12345, batch_size=100000).fit(spatialEnvData2)

# %%
spatialEnvGrid2['clusters_env_spatial2_8'] = km2_8.labels_
spatialEnvGrid2['clusters_env_spatial2_16'] = km2_16.labels_
spatialEnvGrid2['clusters_env_spatial2_22'] = km2_22.labels_
spatialEnvGrid2.to_parquet(folder + "spatialEnvChar2_cluster_grid.pq")

# %%
ax = spatialEnvGrid2.plot('clusters_env_spatial2_8', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction and Spatial Signatures set 2 Clusters Grid (8)")
ax = spatialEnvGrid2.plot('clusters_env_spatial2_16', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction and Spatial Signatures set 2 Clusters Grid (16)")
ax = spatialEnvGrid2.plot('clusters_env_spatial2_22', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction and Spatial Signatures set 2 Clusters Grid (22)")

# %% [markdown]
# ### 3.6. Spatial Signatures
# 
# Final clustering all three components based around the grid layout

# %% [markdown]
# ##### Comparing scores with spatial clustering

# %%
socialTessClustering = gpd.read_parquet(folder + "socialTessClusteringNew.pq")
# socialTessClustering["accessScore"] = centralityAutoCorr["accessScoreNew"]
# socialTessClustering.to_parquet(folder + "socialTessClusteringNew.pq")

# %%
formAoI_gdf = gpd.read_parquet(folder + "convolutions_AoI_usedCharsFinal.pq")
formAoI_gdf.head(1)

# %%
envQual = gpd.read_parquet(folder + "envPredTess.pq")
envQual.head(1)

# %%
medianForm_df = formAoI_gdf.filter(regex='_q2')
medianForm_df.head(1)

# %%
medianForm_df.columns = medianForm_df.columns.str.replace('_q2','')


# %%
avgQual_gdf = medianForm_df.merge(envQual[["heat_score","water_score","geometry"]], left_index=True, right_index=True)
avgQual_gdf = avgQual_gdf.merge(socialTessClustering[["accessScore"]], left_index=True, right_index=True)

# %%
avgQual_gdf

# %%
envScoreData = envQual.drop(columns=["QH","QE","QS","Tsurf","Evap","RO","TotCh","Drainage","SMD","geometry"])
scaler = preprocessing.StandardScaler()
envScoreData = scaler.fit_transform(envScoreData)
envScoreData[np.isnan(envScoreData)] = 0

# %%
socScoreData = socialData2.drop(columns=["clsCnt","btwCnt","strCnt"])
scaler = preprocessing.StandardScaler()
socScore_Data = scaler.fit_transform(socScoreData)
socScore_Data[np.isnan(socScore_Data)] = 0

# %%
socScoreSpatial_data = np.hstack([data,socScore_Data])
envScoreSpatial_data = np.hstack([data,envScoreData])
envSocScoreSpatial_data = np.hstack([data,socScore_Data,envScoreData])
#tessSpatialData

# %%
cg_SpatialSocScore = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_SpatialSocScore.fit(socScoreSpatial_data)

cg_SpatialEnvScore = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_SpatialEnvScore.fit(envScoreSpatial_data)

cg_SpatialEnvSocScore = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_SpatialEnvSocScore.fit(envSocScoreSpatial_data)

# %%
ax1 = cg_SpatialSocScore.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Scores")
ax1.set_xlim(0, 25)
# ax1.axvline(4, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(9, color=c[4], ymin=.02, ymax=.98)


# %%
km_spatial_social = MiniBatchKMeans(n_clusters=9, n_init=1000, random_state=12345, batch_size=100000).fit(socScoreSpatial_data)
# km_spatial_social_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(socScoreSpatial_data)


# %%
tessSpatialData["clusters_spatial_social"] = km_spatial_social.labels_
# tessSpatialData["clusters_spatial_social_4"] = km_spatial_social_4.labels_

# %%
ax1 = cg_SpatialEnvScore.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Environmental Prediction Scores")
ax1.set_xlim(0, 25)
# ax1.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(10, color=c[4], ymin=.02, ymax=.98)

# %%
km_spatial_env_3 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(envScoreSpatial_data)
km_spatial_env_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(envScoreSpatial_data)

# %%
tessSpatialData["clusters_spatial_env"] = km_spatial_env_10.labels_
tessSpatialData["clusters_spatial_env_3"] = km_spatial_env_3.labels_

# %%
ax1 = cg_SpatialEnvSocScore.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social sustainability scores + Environmental Prediction Scores")
ax1.set_xlim(0, 25)
# ax1.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(10, color=c[4], ymin=.02, ymax=.98)

# %%
km_spatial_social_env_3 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(envSocScoreSpatial_data)
km_spatial_social_env_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(envSocScoreSpatial_data)

# %%
km_spatial_social = MiniBatchKMeans(n_clusters=9, n_init=1000, random_state=12345, batch_size=100000).fit(socScoreSpatial_data)
km_spatial_social_env = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(envSocScoreSpatial_data)

# %%
tessSpatialData["clusters_spatial_social_env"] = km_spatial_social_env.labels_
tessSpatialData["clusters_spatial_social"] = km_spatial_social.labels_

# %%
tessClassifying = tessSpatialData.drop(columns=["clusters_spatial_social_4", "clusters_spatial_env_3", "clusters_spatial_social_env_3"])
tessClassifying2 = tessSpatialData.drop(columns=["clusters_spatial_social", "clusters_spatial_env", "clusters_spatial_social_env"])

tessClassifying2.rename(columns={'clusters_spatial_social_4': 'clusters_spatial_social', 
                                 'clusters_spatial_env_3': 'clusters_spatial_env', 
                                 'clusters_spatial_social_env_3': 'clusters_spatial_social_env'}, inplace=True)


# %%
tessSpatialData.to_parquet(folder + "finalClusterForComparisons2.pq")

# %%
# centralityAutoCorr
tessSpatialData = gpd.read_parquet(folder + "finalClusterForComparisons.pq")

# %%
avgQual_gdf["clusters_social"] = tessSpatialData["clusters_social"]
avgQual_gdf["clusters_spatial_social"] = tessSpatialData["clusters_spatial_social"]
avgQual_gdf["clusters_spatial_social_env"] = tessSpatialData["clusters_spatial_social_env"]

# %%
tessSpatialData.head(1)

# %%
avgQual_gdf = avgQual_gdf.merge(tessSpatialData, left_on="geometry", right_on="tessellation")
# tessSpatialData.dissolve('clusters_spatial', aggfunc="mean")
# socialData2
avgQual_gdf = gpd.GeoDataFrame(data=avgQual_gdf.drop(columns=["geometry","tessellation"]), geometry=avgQual_gdf.geometry)
avgQual_gdf.head(1)

# %%
avgQualSpatial_gdf = avgQual_gdf.drop(columns=['clusters_social', 'clusters_env',
       'clusters_spatial_social', 'clusters_spatial_social_4',
       'clusters_spatial_env', 'clusters_spatial_env_3',
       'clusters_spatial_social_env', 'clusters_spatial_social_env_3','buildings'])

# %%
avgQualSpatialDissolve = avgQualSpatial_gdf.dissolve('clusters_spatial', aggfunc="mean")

# %%
avgQualSpatialDissolve.to_parquet(folder + "avgValSpatialDissolveNew.pq")

# %%
avgQualSpatialDissolve

# %%
avgQualSpatialDissolve

# %%
avgQualSpatialDissolveTop = avgQualSpatialDissolve.loc[avgQualSpatialDissolve.index.isin([5,3,6,10,2])]

# %%
avgQualSpatialDissolveTop

# %%
used_characteristics_dict_v2 = {
    'sscCCo': 'circular compactness of ETC', #
    'sicCAR': 'covered area ratio of ETC', # building cover
    'lddNDe': 'local node density of street network',
    'sdsSPO': 'openness of street profile', #
    'ltbIBD': 'mean inter-building distance',
    'mtbAli': 'alignment of neighbouring buildings',
    'lieWCe': 'area-weighted ETCs of enclosure', # 
}

# %%
avgQualSpatialDissolve.to_excel(folder + "avgClusterCharacteristics2.xlsx")

# %%
avgQualSpatialDissolveNorm = avgQualSpatialDissolve[["sscCCo","sicCAR","lddNDe", "sdsSPO","ltbIBD","mtbAli","lieWCe"]].copy()
avgQualSpatialDissolveNorm = (avgQualSpatialDissolveNorm-avgQualSpatialDissolveNorm.min())/(avgQualSpatialDissolveNorm.max()-avgQualSpatialDissolveNorm.min())

# %%
# avgQualSpatialDissolve["sscCCo"].plot.bar()
# avgQualSpatialDissolve["sicCAR"].plot.bar()
# avgQualSpatialDissolve["lddNDe"].plot.bar()
# avgQualSpatialDissolve["sdsSPO"].plot.bar()
# avgQualSpatialDissolve["ltbIBD"].plot.bar()
# avgQualSpatialDissolve["mtbAli"].plot.bar()
# avgQualSpatialDissolve["lieWCe"].plot.bar()

avgQualSpatialCompare = avgQualSpatialDissolveNorm[["sscCCo","sicCAR","lddNDe", "sdsSPO","ltbIBD",
                                    "mtbAli","lieWCe"]].plot.bar(rot=0, figsize=(20,20))
avgQualSpatialCompare.legend([used_characteristics_dict_v2["sscCCo"], used_characteristics_dict_v2["sicCAR"], 
                              used_characteristics_dict_v2["lddNDe"], used_characteristics_dict_v2["sdsSPO"],
                              used_characteristics_dict_v2["ltbIBD"], used_characteristics_dict_v2["mtbAli"], 
                              used_characteristics_dict_v2["lieWCe"]])
avgQualSpatialCompare.set_title("Spatial Signatures of Clusters")
avgQualSpatialCompare.set_xlabel("Cluster Number")
avgQualSpatialCompare.set_ylabel("Normalized Value")

# %%
spatialDissolve = avgQual_gdf.dissolve('clusters_spatial')

# %%
spatialDissolve.to_parquet(folder + "fullDissolveNew.pq")

# %%
ax = tessClassifying.plot('clusters_spatial', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial Clusters")

ax = tessClassifying.set_geometry('buildings').plot('clusters_spatial', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial Clusters Buildings")

ax = tessClassifying.plot('clusters_social', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Interconnectivity Clusters")

ax = tessClassifying.set_geometry('buildings').plot('clusters_social', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Interconnectivity Clusters Buildings")

ax = tessClassifying.plot('clusters_env', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmental Prediction Clusters")

ax = tessClassifying.set_geometry('buildings').plot('clusters_env', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmental Prediction Clusters Buildings")

ax = tessClassifying.plot('clusters_spatial_social', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial + Interconnectivity Clusters")

ax = tessClassifying.set_geometry('buildings').plot('clusters_spatial_social', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial + Interconnectivity Clusters Buildings")

ax = tessClassifying.plot('clusters_spatial_env', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial + Environmental Prediction Clusters")

ax = tessClassifying.set_geometry('buildings').plot('clusters_spatial_env', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial + Environmental Prediction Clusters Buildings")

ax = tessClassifying.plot('clusters_spatial_social_env', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial + Interconnectivity + Environmental Prediction Clusters")

ax = tessClassifying.set_geometry('buildings').plot('clusters_spatial_social_env', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial + Interconnectivity + Environmental Prediction Clusters Buildings")

# %%
val = 0
for cols in tessClassifying.columns:
    if cols == "buildings" or cols == "tessellation":
        continue
    val += 1
    tessClassifying[cols] += 100*val
    tessClassifying2[cols] += 100*val

# %%
tessClassifying2.head(5)

# %%
tessClassifyer = tessClassifying.drop(columns=["buildings","tessellation"])

# %%
tessClassifying2.head(1)

# %%
probNames = tessClassifyer.columns.to_list()
probNames.pop(0)

mat_soc_spat_size = (11,5)
mat_env_spat_size = (11,4)
mat_socSpat_spat_size = (11,9)
mat_envSpat_spat_size = (11,10)
mat_socEnvSpat_spat_size = (11,10)
social_likelihood_spatial = np.zeros(mat_soc_spat_size)
env_likelihood_spatial = np.zeros(mat_env_spat_size)
socSpat_likelihood_spatial = np.zeros(mat_socSpat_spat_size)
envSpat_likelihood_spatial = np.zeros(mat_envSpat_spat_size)
socEnvSpat_likelihood_spatial = np.zeros(mat_socEnvSpat_spat_size)

probDict = dict(zip(probNames, [social_likelihood_spatial, env_likelihood_spatial, socSpat_likelihood_spatial, envSpat_likelihood_spatial,
                                socEnvSpat_likelihood_spatial]))

# %%
probDict2 = dict(zip(probNames, [social_likelihood_spatial, env_likelihood_spatial, socSpat_likelihood_spatial, envSpat_likelihood_spatial,
                                socEnvSpat_likelihood_spatial]))

# %%
spatialTemp = tessClassifyer.loc[tessClassifyer["clusters_spatial"] == 0]

# %%
len(tessClassifyer.loc[tessClassifyer["clusters_social"]==0])

# %%
spatialClustLen = len(tessClassifyer["clusters_spatial"].unique())
spatIter = iter(range(spatialClustLen))
# currSpat = 0
# while currSpat < spatialClustLen:
for i in spatIter:
    spatialTemp = tessClassifyer.loc[tessClassifyer["clusters_spatial"] == i]   
    for cols in tessClassifyer.columns:
        if cols == "clusters_spatial":
            continue
        clustNum = len(tessClassifyer[cols].unique())
        probMat = probDict[cols]
        # currClust = 0 
        currIter = iter(range(clustNum))
        # while currClust < clustNum:
        for j in currIter:
            clustOI = spatialTemp.loc[tessClassifyer[cols] == j]
            clustTotal = tessClassifyer.loc[tessClassifyer[cols] == j]
            clustProb = len(clustOI)/len(clustTotal)
            probMat[i,j] = clustProb

        probDict[cols] = probMat

    

# %%
cluster_index_max_overlap = {}
for col in probNames:
    cluster_index_max_overlap[col] = np.argmax(probDict[col], axis=1)

# %%
cluster_index_max_overlap_DF = pd.DataFrame.from_dict(cluster_index_max_overlap)
cluster_index_max_overlap_DF.to_excel(folder + "spatialClusterOverlapIndex2.xlsx")

# %%
vv = probDict2[probNames[4]]
np.sum(vv[:,4])


# %%
len(probDict[probNames[0]].T)

# %%
expandedDict2 = {}
for col in probNames:
    numCols = len(probDict[col].T)
    val = iter(range(numCols))
    currMat = probDict[col]
    for v in val:
        colName = col + "_" + str(v)
        expandedDict2[colName] = currMat[:,v]

# %%
probDF2 = pd.DataFrame.from_dict(expandedDict2)

# %%
probDF2.to_excel(folder + "spatialClusterOverlapScoresAllFinal2.xlsx")

# %%
overlap_scores = {}
overlapScoreExp = {}
for col in probNames:
    overlap_scores[col] = np.max(probDict[col], axis=1)
    overlapScoreExp[col] = np.sum(overlap_scores[col])/len(overlap_scores[col])

# %%
overlapDF = pd.DataFrame.from_dict(overlap_scores)

# %%
overlapDF.to_excel(folder + "spatialClusterOverlapScores2.xlsx")

# %%
overlapScoreExp

# %% [markdown]
# ##### Grid computation (no longer relevant)

# %%
socialChars = ["accessScore", "amenityDiversity", "HOUSING20", "POP20", "housing_score"]

# %%
socialGrid = area_interpolate(socialTessClustering, spatialEnvGrid1, socialChars, n_jobs=1)

# %%
spatialEnvSocialGrid1 = spatialEnvGrid1.drop(columns=["clusters_env_spatial1_13", 
                                                      "clusters_env_spatial1_19", 
                                                      "clusters_env_spatial1_21"]).merge(socialGrid, on="geometry")
spatialEnvSocialGrid2 = spatialEnvGrid2.drop(columns=["clusters_env_spatial2_8", 
                                                      "clusters_env_spatial2_16", 
                                                      "clusters_env_spatial2_22"]).merge(socialGrid, on="geometry")

# %% [markdown]
# ##### Tesselation clustering (no longer relevant)

# %%
#combined convolution data
formAoI_gdf = gpd.read_parquet(folder + "convolutions_AoI_usedCharsFinal.pq")
# formAoI_gdf1 = gpd.read_parquet(folder + "convolutions_AoI_usedCharsFullFinal.pq")
# formAoI_gdf2 = gpd.read_parquet(folder + "convolutions_AoI_usedCharsFinal.pq")
formAoI_gdf.head(1)

# %%
social_data_gdf = gpd.read_parquet(folder + "socialTessData.pq")

# %%
social_data_gdf.drop(columns=["housing_score"], inplace=True)
social_data_gdf.head(1)

# %%
envPredTess = gpd.read_parquet(folder + "envPredTess.pq")

# %%
envPredTess.head(1)

# %%
spatialEnvSocialTess = formAoI_gdf.merge(socialTessClustering.drop(columns=["tessellation", "housing_score"]), on="tID")
# spatialEnvSocial2Tess = formAoI_gdf.merge(socialTessClustering.drop(columns=["tessellation"]), on="tID")
spatialEnvSocialTess = spatialEnvSocialTess.merge(envPredTess, on="geometry")
# spatialEnvSocialTess2 = spatialEnvSocialTess2.merge(envPredTess, on="geometry")

# %%
pd.set_option('display.max_columns', None)
spatialEnvSocialTess.head(1)

# %%
used_characteristics_v2 = {
    'sscCCo', # 'circular compactness of ETC', #
    'sicCAR', # 'covered area ratio of ETC', # building cover
    'mtbAli', # 'alignment of neighbouring buildings',
    'ltbIBD', # 'mean inter-building distance',
    'sdsSPO', # 'openness of street profile', #
    'lddNDe', # 'local node density of street network',
    'lieWCe', # 'area-weighted ETCs of enclosure', # 
}

# %% [markdown]
# ##### Data prep to find final sets of data

# %% [markdown]
# Clustering with all env categories and different social categories

# %%
scaler = preprocessing.StandardScaler()
dataTessAll = scaler.fit_transform(spatialEnvSocialTess.drop(columns=["geometry"]))
dataTessAll[np.isnan(dataTessAll)] = 0

# No Housing or Population data
scaler = preprocessing.StandardScaler()
dataTessNoPop = scaler.fit_transform(spatialEnvSocialTess.drop(columns=["HOUSING20",
                                                                        "POP20","geometry"]))
dataTessNoPop[np.isnan(dataTessNoPop)] = 0

# No Access score
scaler = preprocessing.StandardScaler()
dataTessNoSocScore = scaler.fit_transform(spatialEnvSocialTess.drop(columns=["accessScore",
                                                                             "geometry"]))
dataTessNoSocScore[np.isnan(dataTessNoSocScore)] = 0

# No Access or Housing score
scaler = preprocessing.StandardScaler()
dataTessNoSocScoreNoH = scaler.fit_transform(spatialEnvSocialTess.drop(columns=["accessScore",
                                                                             "HOUSING20",
                                                                             "geometry"]))
dataTessNoSocScoreNoH[np.isnan(dataTessNoSocScoreNoH)] = 0

# No Access or Housing score or Population Data
scaler = preprocessing.StandardScaler()
dataTessOnlyConn = scaler.fit_transform(spatialEnvSocialTess.drop(columns=["accessScore",
                                                                             "HOUSING20",
                                                                             "POP20",
                                                                             "geometry"]))
dataTessOnlyConn[np.isnan(dataTessOnlyConn)] = 0


# %%
envCharsCols = ["QH","QE","QS","Tsurf","Evap","RO","TotCh","Drainage","SMD","geometry"]
envScoresCols = ["heat_score","water_score","geometry"]


# %%
spatialEnvSocialTess.head(1)

# %%
buildingAlighment = ["mtbAli_q1", "mtbAli_q2", "mtbAli_q3"] 
areaWeightedETC = ["lieWCe_q1", "lieWCe_q2", "lieWCe_q3"]
streetOpenness = ["sdsSPO_q1", "sdsSPO_q2", "sdsSPO_q3"]


# %%
connectivityCols = ["clsCnt","btwCnt","strCnt"]

# %% [markdown]
# All varieties of social sustainability clustering for just environmental scores

# %%
scaler = preprocessing.StandardScaler()
dataTessEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=envCharsCols))
dataTessEnvScores[np.isnan(dataTessEnvScores)] = 0

# No Housing or Population data
scaler = preprocessing.StandardScaler()
noPop = ["HOUSING20","POP20"]
envCharsColsNoPop = envCharsCols + noPop
dataTessNoPopEnvScores = scaler.fit_transform(spatialEnvSocialTess
                                              .drop(columns=envCharsColsNoPop))
dataTessNoPopEnvScores[np.isnan(dataTessNoPopEnvScores)] = 0

# No Access score
scaler = preprocessing.StandardScaler()
noAccScore = ["accessScore"]
envCharsColsNoAccScore = envCharsCols + noAccScore
dataTessNoSocScoreEnvScores = scaler.fit_transform(spatialEnvSocialTess
                                                   .drop(columns=envCharsColsNoAccScore))
dataTessNoSocScoreEnvScores[np.isnan(dataTessNoSocScoreEnvScores)] = 0

# No Access or Housing score
scaler = preprocessing.StandardScaler()
noAccScoreNoH = ["accessScore","HOUSING20"]
envCharsColsNoAccScore = envCharsCols + noAccScoreNoH
dataTessNoSocScoreNoHEnvScores = scaler.fit_transform(spatialEnvSocialTess
                                                      .drop(columns=envCharsColsNoAccScore))
dataTessNoSocScoreNoHEnvScores[np.isnan(dataTessNoSocScoreNoHEnvScores)] = 0



# No Access or Housing score or population data
scaler = preprocessing.StandardScaler()
onlyConn = envCharsCols + noPop + noAccScore
dataTessOnlyConnEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTessOnlyConnEnvScores[np.isnan(dataTessOnlyConnEnvScores)] = 0

# %%


# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = envCharsCols + noPop + connectivityCols
dataTessOnlyAccEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTessOnlyAccEnvScores[np.isnan(dataTessOnlyAccEnvScores)] = 0



# %% [markdown]
# All social sustainability clustering options for environmental prediction characters

# %%
scaler = preprocessing.StandardScaler()
dataTessEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=envScoresCols))
dataTessEnvChars[np.isnan(dataTessEnvChars)] = 0

# No Housing or Population data
scaler = preprocessing.StandardScaler()
noPop = ["HOUSING20","POP20"]
envScoresColsNoPop = envScoresCols + noPop
dataTessNoPopEnvChars = scaler.fit_transform(spatialEnvSocialTess
                                             .drop(columns=envScoresColsNoPop))
dataTessNoPopEnvChars[np.isnan(dataTessNoPopEnvChars)] = 0

# No Access score
scaler = preprocessing.StandardScaler()
noAccScore = ["accessScore"]
envScoresColsNoAccScore = envScoresCols + noAccScore
dataTessNoSocScoreEnvChars = scaler.fit_transform(spatialEnvSocialTess
                                                  .drop(columns=envScoresColsNoAccScore))
dataTessNoSocScoreEnvChars[np.isnan(dataTessNoSocScoreEnvChars)] = 0

# No Access or Housing score
scaler = preprocessing.StandardScaler()
noAccScoreNoH = ["accessScore","HOUSING20"]
envScoresColsNoAccScore = envScoresCols + noAccScoreNoH
dataTessNoSocScoreNoHEnvChars = scaler.fit_transform(spatialEnvSocialTess
                                                     .drop(columns=envScoresColsNoAccScore))
dataTessNoSocScoreNoHEnvChars[np.isnan(dataTessNoSocScoreNoHEnvChars)] = 0


# No Access or Housing score or Population data
scaler = preprocessing.StandardScaler()
onlyConn = envScoresCols + noPop + noAccScore
dataTessOnlyConnEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTessOnlyConnEnvChars[np.isnan(dataTessOnlyConnEnvChars)] = 0

# %%


# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = envScoresCols + noPop + connectivityCols
dataTessOnlyAccEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTessOnlyAccEnvChars[np.isnan(dataTessOnlyAccEnvChars)] = 0

# %% [markdown]
# ##### Final Set of data descriptors

# %%
used_characteristics_v2 = {
    'sscCCo', # 'circular compactness of ETC', #
    'sicCAR', # 'covered area ratio of ETC', # building cover
    'mtbAli', # 'alignment of neighbouring buildings',
    'ltbIBD', # 'mean inter-building distance',
    'sdsSPO', # 'openness of street profile', #
    'lddNDe', # 'local node density of street network',
    'lieWCe', # 'area-weighted ETCs of enclosure', # 
}

# %%
formAoI_gdf = gpd.read_parquet(folder + "convolutions_AoI_usedCharsFinal.pq")
social_data_gdf = gpd.read_parquet(folder + "socialTessData.pq")
envPredTess = gpd.read_parquet(folder + "envPredTess.pq")

spatialEnvSocialTess = formAoI_gdf.merge(social_data_gdf.drop(columns=["tessellation", "housing_score"]), on="tID")
spatialEnvSocialTess = spatialEnvSocialTess.merge(envPredTess, on="geometry")

# %%
buildingAlighment = ["mtbAli_q1", "mtbAli_q2", "mtbAli_q3"] 
meanBuildingDist = ["ltbIBD_q1", "ltbIBD_q2", "ltbIBD_q3"]
areaWeightedETC = ["lieWCe_q1", "lieWCe_q2", "lieWCe_q3"]
streetOpenness = ["sdsSPO_q1", "sdsSPO_q2", "sdsSPO_q3"]

spatialComb1 = buildingAlighment + meanBuildingDist
spatialComb2 = spatialComb1 + areaWeightedETC
# spatialComb2 = buildingAlighment + areaWeightedETC
# spatialComb3 = buildingAlighment + streetOpenness
# spatialComb4 = meanBuildingDist + streetOpenness
# spatialComb5 = meanBuildingDist + areaWeightedETC

envCharsCols = ["QH","QE","QS","Tsurf","Evap","RO","TotCh","Drainage","SMD","geometry"]
envScoresCols = ["heat_score","water_score","geometry"]
geom = ["geometry"]

noPop = ["HOUSING20","POP20"]
accScore = ["accessScore"]
connectivityCols = ["clsCnt","btwCnt","strCnt"]

onlyConn = noPop + accScore
onlyAccess = noPop + connectivityCols

# Spatial All
onlyConnEnvAllSpatialAll = onlyConn + geom
onlyAccEnvAllSpatialAll = onlyAccess + geom

onlyConnEnvScoreSpatialAll = onlyConn + envCharsCols
onlyAccEnvScoreSpatialAll = onlyAccess + envCharsCols

onlyConnEnvCharSpatialAll = onlyConn + envScoresCols
onlyAccEnvCharSpatialAll = onlyAccess + envScoresCols

# Spatial Combo 1
onlyConnEnvAllSpatial1 = onlyConnEnvAllSpatialAll + spatialComb1
onlyAccEnvAllSpatial1 = onlyAccEnvAllSpatialAll + spatialComb1

onlyConnEnvScoreSpatial1 = onlyConnEnvScoreSpatialAll + spatialComb1
onlyAccEnvScoreSpatial1 = onlyAccEnvScoreSpatialAll + spatialComb1

onlyConnEnvCharSpatial1 = onlyConnEnvCharSpatialAll + spatialComb1
onlyAccEnvCharSpatial1 = onlyAccEnvCharSpatialAll + spatialComb1

# Spatial Combo 2
onlyConnEnvAllSpatial2 = onlyConnEnvAllSpatialAll + spatialComb2
onlyAccEnvAllSpatial2 = onlyAccEnvAllSpatialAll + spatialComb2

onlyConnEnvScoreSpatial2 = onlyConnEnvScoreSpatialAll + spatialComb2
onlyAccEnvScoreSpatial2 = onlyAccEnvScoreSpatialAll + spatialComb2

onlyConnEnvCharSpatial2 = onlyConnEnvCharSpatialAll + spatialComb2
onlyAccEnvCharSpatial2 = onlyAccEnvCharSpatialAll + spatialComb2

# %%
# Env All
# All spatial
# No Access score, Housing data, or Population Data
scaler = preprocessing.StandardScaler()
dataTessSpatialAllOnlyConn = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConnEnvAllSpatialAll))
dataTessSpatialAllOnlyConn[np.isnan(dataTessSpatialAllOnlyConn)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
dataTessSpatialAllOnlyAcc = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccEnvAllSpatialAll))
dataTessSpatialAllOnlyAcc[np.isnan(dataTessSpatialAllOnlyAcc)] = 0

# spatial combo 1
# No Access score, Housing data, or Population Data
scaler = preprocessing.StandardScaler()
dataTessSpatial1OnlyConn = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConnEnvAllSpatial1))
dataTessSpatial1OnlyConn[np.isnan(dataTessSpatial1OnlyConn)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
dataTessSpatial1OnlyAcc = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccEnvAllSpatial1))
dataTessSpatial1OnlyAcc[np.isnan(dataTessSpatial1OnlyAcc)] = 0

# spatial combo 2
# No Access score, Housing data, or Population Data
scaler = preprocessing.StandardScaler()
dataTessSpatial2OnlyConn = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConnEnvAllSpatial2))
dataTessSpatial2OnlyConn[np.isnan(dataTessSpatial2OnlyConn)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
dataTessSpatial2OnlyAcc = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccEnvAllSpatial2))
dataTessSpatial2OnlyAcc[np.isnan(dataTessSpatial2OnlyAcc)] = 0

# %%
# Env Score
# No Access or Housing score or population data
scaler = preprocessing.StandardScaler()
dataTessOnlyConnEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConnEnvScoreSpatialAll))
dataTessOnlyConnEnvScores[np.isnan(dataTessOnlyConnEnvScores)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
dataTessOnlyAccEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccEnvScoreSpatialAll))
dataTessOnlyAccEnvScores[np.isnan(dataTessOnlyAccEnvScores)] = 0

# Spatial Combo 1
# No Access or Housing score or population data
scaler = preprocessing.StandardScaler()
dataTess1OnlyConnEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConnEnvScoreSpatial1))
dataTess1OnlyConnEnvScores[np.isnan(dataTess1OnlyConnEnvScores)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
dataTess1OnlyAccEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccEnvScoreSpatial1))
dataTess1OnlyAccEnvScores[np.isnan(dataTess1OnlyAccEnvScores)] = 0

# Spatial Combo 2
# No Access or Housing score or population data
scaler = preprocessing.StandardScaler()
dataTess2OnlyConnEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConnEnvScoreSpatial2))
dataTess2OnlyConnEnvScores[np.isnan(dataTess2OnlyConnEnvScores)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
dataTess2OnlyAccEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccEnvScoreSpatial2))
dataTess2OnlyAccEnvScores[np.isnan(dataTess2OnlyAccEnvScores)] = 0


# %%
# Env Char
# All spatial
# No Access or Housing score or Population data
scaler = preprocessing.StandardScaler()
dataTessOnlyConnEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConnEnvCharSpatialAll))
dataTessOnlyConnEnvChars[np.isnan(dataTessOnlyConnEnvChars)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
dataTessOnlyAccEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccEnvCharSpatialAll))
dataTessOnlyAccEnvChars[np.isnan(dataTessOnlyAccEnvChars)] = 0

# spatial combo 1
# No Access or Housing score or Population data
scaler = preprocessing.StandardScaler()
dataTess1OnlyConnEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConnEnvCharSpatial1))
dataTess1OnlyConnEnvChars[np.isnan(dataTess1OnlyConnEnvChars)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
dataTess1OnlyAccEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccEnvCharSpatial1))
dataTess1OnlyAccEnvChars[np.isnan(dataTess1OnlyAccEnvChars)] = 0

# spatial combo 2
# No Access or Housing score or Population data
scaler = preprocessing.StandardScaler()
dataTess2OnlyConnEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConnEnvCharSpatial2))
dataTess2OnlyConnEnvChars[np.isnan(dataTess2OnlyConnEnvChars)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
dataTess2OnlyAccEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccEnvCharSpatial2))
dataTess2OnlyAccEnvChars[np.isnan(dataTess2OnlyAccEnvChars)] = 0


# %% [markdown]
# ###### Previous Finalized data prep

# %%
# Env All
# spatial combo 1
# No Access score, Housing data, or Population Data
scaler = preprocessing.StandardScaler()
onlyConn = noPop + noAccScore + spatialComb1 + geom
dataTessSpatial1OnlyConn = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTessSpatial1OnlyConn[np.isnan(dataTessSpatial1OnlyConn)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = noPop + connectivityCols + spatialComb1 + geom
dataTessSpatial1OnlyAcc = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTessSpatial1OnlyAcc[np.isnan(dataTessSpatial1OnlyAcc)] = 0

# spatial combo 2
# No Access score, Housing data, or Population Data
scaler = preprocessing.StandardScaler()
onlyConn = noPop + noAccScore + spatialComb2 + geom
dataTessSpatial2OnlyConn = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTessSpatial2OnlyConn[np.isnan(dataTessSpatial2OnlyConn)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = noPop + connectivityCols + spatialComb2 + geom
dataTessSpatial2OnlyAcc = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTessSpatial2OnlyAcc[np.isnan(dataTessSpatial2OnlyAcc)] = 0

# spatial combo 3
# No Access score, Housing data, or Population Data
scaler = preprocessing.StandardScaler()
onlyConn = noPop + noAccScore + spatialComb3 + geom
dataTessSpatial3OnlyConn = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTessSpatial3OnlyConn[np.isnan(dataTessSpatial3OnlyConn)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = noPop + connectivityCols + spatialComb3 + geom
dataTessSpatial3OnlyAcc = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTessSpatial3OnlyAcc[np.isnan(dataTessSpatial3OnlyAcc)] = 0

# spatial combo 4
# No Access score, Housing data, or Population Data
scaler = preprocessing.StandardScaler()
onlyConn = noPop + noAccScore + spatialComb4 + geom
dataTessSpatial4OnlyConn = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTessSpatial4OnlyConn[np.isnan(dataTessSpatial4OnlyConn)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = noPop + connectivityCols + spatialComb4 + geom
dataTessSpatial4OnlyAcc = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTessSpatial4OnlyAcc[np.isnan(dataTessSpatial4OnlyAcc)] = 0

# spatial combo 5
# No Access score, Housing data, or Population Data
scaler = preprocessing.StandardScaler()
onlyConn = noPop + noAccScore + spatialComb5 + geom
dataTessSpatial5OnlyConn = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTessSpatial5OnlyConn[np.isnan(dataTessSpatial5OnlyConn)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = noPop + connectivityCols + spatialComb5 + geom
dataTessSpatial5OnlyAcc = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTessSpatial5OnlyAcc[np.isnan(dataTessSpatial5OnlyAcc)] = 0

# %%
# Env Score
# No Access or Housing score or population data
scaler = preprocessing.StandardScaler()
onlyConn = envCharsCols + noPop + noAccScore
dataTessOnlyConnEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTessOnlyConnEnvScores[np.isnan(dataTessOnlyConnEnvScores)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = envCharsCols + noPop + connectivityCols
dataTessOnlyAccEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTessOnlyAccEnvScores[np.isnan(dataTessOnlyAccEnvScores)] = 0

# Spatial Combo 1
# No Access or Housing score or population data
scaler = preprocessing.StandardScaler()
onlyConn = envCharsCols + noPop + noAccScore + spatialComb1
dataTess1OnlyConnEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTess1OnlyConnEnvScores[np.isnan(dataTess1OnlyConnEnvScores)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = envCharsCols + noPop + connectivityCols + spatialComb1
dataTess1OnlyAccEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTess1OnlyAccEnvScores[np.isnan(dataTess1OnlyAccEnvScores)] = 0

# Spatial Combo 2
# No Access or Housing score or population data
scaler = preprocessing.StandardScaler()
onlyConn = envCharsCols + noPop + noAccScore + spatialComb2
dataTess2OnlyConnEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTess2OnlyConnEnvScores[np.isnan(dataTess2OnlyConnEnvScores)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = envCharsCols + noPop + connectivityCols + spatialComb2
dataTess2OnlyAccEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTess2OnlyAccEnvScores[np.isnan(dataTess2OnlyAccEnvScores)] = 0

# Spatial Combo 3
# No Access or Housing score or population data
scaler = preprocessing.StandardScaler()
onlyConn = envCharsCols + noPop + noAccScore + spatialComb3
dataTess3OnlyConnEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTess3OnlyConnEnvScores[np.isnan(dataTess3OnlyConnEnvScores)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = envCharsCols + noPop + connectivityCols + spatialComb3
dataTess3OnlyAccEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTess3OnlyAccEnvScores[np.isnan(dataTess3OnlyAccEnvScores)] = 0

# Spatial Combo 4
# No Access or Housing score or population data
scaler = preprocessing.StandardScaler()
onlyConn = envCharsCols + noPop + noAccScore + spatialComb4
dataTess4OnlyConnEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTess4OnlyConnEnvScores[np.isnan(dataTess4OnlyConnEnvScores)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = envCharsCols + noPop + connectivityCols + spatialComb4
dataTess4OnlyAccEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTess4OnlyAccEnvScores[np.isnan(dataTess4OnlyAccEnvScores)] = 0

# Spatial Combo 5
# No Access or Housing score or population data
scaler = preprocessing.StandardScaler()
onlyConn = envCharsCols + noPop + noAccScore + spatialComb5
dataTess5OnlyConnEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTess5OnlyConnEnvScores[np.isnan(dataTess5OnlyConnEnvScores)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = envCharsCols + noPop + connectivityCols + spatialComb5
dataTess5OnlyAccEnvScores = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTess5OnlyAccEnvScores[np.isnan(dataTess5OnlyAccEnvScores)] = 0


# %%
# Env Char
# All spatial
# No Access or Housing score or Population data
scaler = preprocessing.StandardScaler()
onlyConn = envScoresCols + noPop + noAccScore
dataTessOnlyConnEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTessOnlyConnEnvChars[np.isnan(dataTessOnlyConnEnvChars)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = envScoresCols + noPop + connectivityCols
dataTessOnlyAccEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTessOnlyAccEnvChars[np.isnan(dataTessOnlyAccEnvChars)] = 0

# spatial combo 1
# No Access or Housing score or Population data
scaler = preprocessing.StandardScaler()
onlyConn = envScoresCols + noPop + noAccScore + spatialComb1
dataTess1OnlyConnEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTess1OnlyConnEnvChars[np.isnan(dataTess1OnlyConnEnvChars)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = envScoresCols + noPop + connectivityCols + spatialComb1
dataTess1OnlyAccEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTess1OnlyAccEnvChars[np.isnan(dataTess1OnlyAccEnvChars)] = 0

# spatial combo 2
# No Access or Housing score or Population data
scaler = preprocessing.StandardScaler()
onlyConn = envScoresCols + noPop + noAccScore + spatialComb2
dataTess2OnlyConnEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTess2OnlyConnEnvChars[np.isnan(dataTess2OnlyConnEnvChars)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = envScoresCols + noPop + connectivityCols + spatialComb2
dataTess2OnlyAccEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTess2OnlyAccEnvChars[np.isnan(dataTess2OnlyAccEnvChars)] = 0

# spatial combo 3
# No Access or Housing score or Population data
scaler = preprocessing.StandardScaler()
onlyConn = envScoresCols + noPop + noAccScore + spatialComb3
dataTess3OnlyConnEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTess3OnlyConnEnvChars[np.isnan(dataTess3OnlyConnEnvChars)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = envScoresCols + noPop + connectivityCols + spatialComb3
dataTess3OnlyAccEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTess3OnlyAccEnvChars[np.isnan(dataTess3OnlyAccEnvChars)] = 0

# spatial combo 4
# No Access or Housing score or Population data
scaler = preprocessing.StandardScaler()
onlyConn = envScoresCols + noPop + noAccScore + spatialComb4
dataTess4OnlyConnEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTess4OnlyConnEnvChars[np.isnan(dataTess4OnlyConnEnvChars)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = envScoresCols + noPop + connectivityCols + spatialComb4
dataTess4OnlyAccEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTess4OnlyAccEnvChars[np.isnan(dataTess4OnlyAccEnvChars)] = 0

# spatial combo 5
# No Access or Housing score or Population data
scaler = preprocessing.StandardScaler()
onlyConn = envScoresCols + noPop + noAccScore + spatialComb5
dataTess5OnlyConnEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyConn))
dataTess5OnlyConnEnvChars[np.isnan(dataTess5OnlyConnEnvChars)] = 0

# No Housing or population data or connectivity indicators
scaler = preprocessing.StandardScaler()
onlyAccess = envScoresCols + noPop + connectivityCols + spatialComb5
dataTess5OnlyAccEnvChars = scaler.fit_transform(spatialEnvSocialTess.drop(columns=onlyAccess))
dataTess5OnlyAccEnvChars[np.isnan(dataTess5OnlyAccEnvChars)] = 0

# %% [markdown]
# ##### Older data prep

# %%
spatialEnvSocialTess3 = formAoI_gdf2.merge(socialTessClustering.drop(columns=["tessellation","HOUSING20","POP20"]), on="tID")
spatialEnvSocialTess3 = spatialEnvSocialTess3.merge(envPredTess, on="geometry")
spatialEnvSocialTess4 = formAoI_gdf2.merge(socialTessClustering.drop(columns=["tessellation","amenityDiversity","HOUSING20","POP20"]), on="tID")
spatialEnvSocialTess4 = spatialEnvSocialTess4.merge(envPredTess, on="geometry")
spatialEnvSocialTess5 = formAoI_gdf2.merge(socialTessClustering.drop(columns=["tessellation","amenityDiversity","HOUSING20","POP20","housing_score"]), on="tID")
spatialEnvSocialTess5 = spatialEnvSocialTess5.merge(envPredTess, on="geometry")
spatialEnvSocialTess6 = spatialEnvSocialTess5.drop(columns=["heat_score", "water_score"])
spatialEnvSocialTess7 = spatialEnvSocialTess5.drop(columns=["QH", "QE","QS","Tsurf", "Evap", "RO", "TotCh", "Drainage", "SMD"])
spatialEnvSocialTess8 = spatialEnvSocialTess7.drop(columns=["mtbAli_q1", "mtbAli_q2", "mtbAli_q3", "lieWCe_q1", "lieWCe_q2", "lieWCe_q3"])
spatialEnvSocialTess9 = spatialEnvSocialTess7.drop(columns=["mtbAli_q1", "mtbAli_q2", "mtbAli_q3", "sdsSPO_q1", "sdsSPO_q2", "sdsSPO_q3"])
spatialEnvSocialTess10 = spatialEnvSocialTess6.drop(columns=["mtbAli_q1", "mtbAli_q2", "mtbAli_q3", "sdsSPO_q1", "sdsSPO_q2", "sdsSPO_q3"])

# %%
scaler = preprocessing.StandardScaler()
dataTess1 = scaler.fit_transform(spatialEnvSocialTess1.drop(columns=["geometry"]))
dataTess1[np.isnan(dataTess1)] = 0
scaler = preprocessing.StandardScaler()
dataTess2 = scaler.fit_transform(spatialEnvSocialTess2.drop(columns=["geometry"]))
dataTess2[np.isnan(dataTess2)] = 0
scaler = preprocessing.StandardScaler()
dataTess3 = scaler.fit_transform(spatialEnvSocialTess3.drop(columns=["geometry"]))
dataTess3[np.isnan(dataTess3)] = 0
scaler = preprocessing.StandardScaler()
dataTess4 = scaler.fit_transform(spatialEnvSocialTess4.drop(columns=["geometry"]))
dataTess4[np.isnan(dataTess4)] = 0
scaler = preprocessing.StandardScaler()
dataTess5 = scaler.fit_transform(spatialEnvSocialTess5.drop(columns=["geometry"]))
dataTess5[np.isnan(dataTess5)] = 0
scaler = preprocessing.StandardScaler()
dataTess6 = scaler.fit_transform(spatialEnvSocialTess6.drop(columns=["geometry"]))
dataTess6[np.isnan(dataTess6)] = 0
scaler = preprocessing.StandardScaler()
dataTess7 = scaler.fit_transform(spatialEnvSocialTess7.drop(columns=["geometry"]))
dataTess7[np.isnan(dataTess7)] = 0
scaler = preprocessing.StandardScaler()
dataTess8 = scaler.fit_transform(spatialEnvSocialTess8.drop(columns=["geometry"]))
dataTess8[np.isnan(dataTess8)] = 0
scaler = preprocessing.StandardScaler()
dataTess9 = scaler.fit_transform(spatialEnvSocialTess9.drop(columns=["geometry"]))
dataTess9[np.isnan(dataTess9)] = 0
scaler = preprocessing.StandardScaler()
dataTess10 = scaler.fit_transform(spatialEnvSocialTess10.drop(columns=["geometry"]))
dataTess10[np.isnan(dataTess10)] = 0

# %% [markdown]
# #### Cluster Analysis Final

# %% [markdown]
# ##### Env Pred All Cluster analysis

# %%
cg_SpatialAllOnlyConn = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_SpatialAllOnlyConn.fit(dataTessSpatialAllOnlyConn)

cg_SpatialAllOnlyAcc = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_SpatialAllOnlyAcc.fit(dataTessSpatialAllOnlyAcc)

cg_Spatial1OnlyConn = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial1OnlyConn.fit(dataTessSpatial1OnlyConn)

cg_Spatial1OnlyAcc = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial1OnlyAcc.fit(dataTessSpatial1OnlyAcc)

cg_Spatial2OnlyConn = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial2OnlyConn.fit(dataTessSpatial2OnlyConn)

cg_Spatial2OnlyAcc = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial2OnlyAcc.fit(dataTessSpatial2OnlyAcc)



# %%
ax1 = cg_SpatialAllOnlyConn.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (All selected) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (all) Silhouette score")
ax1.set_xlim(0, 25)


# %%
ax1 = cg_SpatialAllOnlyConn.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (All selected) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (all) Silhouette score")
ax1.set_xlim(0, 25)
ax1.axvline(4, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(11, color=c[4], ymin=.02, ymax=.98)


# %%
c = ugg.COLORS
ax = cg_SpatialAllOnlyConn.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)


plt.savefig(folder + "cgramSpatialAllOnlyConn.png", dpi=300, bbox_inches="tight")


# %%
c = ugg.COLORS
ax = cg_SpatialAllOnlyConn.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
ax.axvline(5, color=c[4], ymin=.02, ymax=.98)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
cg_SpatialAllOnlyAcc = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_SpatialAllOnlyAcc.fit(dataTessSpatialAllOnlyAcc)

ax2 = cg_SpatialAllOnlyAcc.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (All selected) + Social Sustainability Indicators (only access score) + Environmental Prediction (all) Silhouette score")
ax2.set_xlim(0, 25)


# %%
ax2 = cg_SpatialAllOnlyAcc.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (All selected) + Social Sustainability Indicators (only access score) + Environmental Prediction (all) Silhouette score")
ax2.set_xlim(0, 25)
ax2.axvline(4, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
c = ugg.COLORS
ax = cg_SpatialAllOnlyAcc.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatialAllOnlyAcc.png", dpi=300, bbox_inches="tight")


# %%
c = ugg.COLORS
ax = cg_SpatialAllOnlyAcc.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
ax.axvline(5, color=c[4], ymin=.02, ymax=.98)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# Above clustergram is easy to differentiate the different component separations. Thus even if the silhouette scores are not the highest, it still seems meaningful.

# %%
ax1 = cg_Spatial1OnlyConn.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (all) Silhouette score")
ax1.set_xlim(0, 25)


# %%
ax1 = cg_Spatial1OnlyConn.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (all) Silhouette score")
ax1.set_xlim(0, 25)
ax1.axvline(5, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
c = ugg.COLORS
ax = cg_Spatial1OnlyConn.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)

plt.savefig(folder + "cgramSpatial1OnlyConn.png", dpi=300, bbox_inches="tight")


# %%
c = ugg.COLORS
ax = cg_Spatial1OnlyConn.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
ax.axvline(10, color=c[4], ymin=.02, ymax=.98)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
ax2 = cg_Spatial1OnlyAcc.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only access score) + Environmental Prediction (all) Silhouette score")
ax2.set_xlim(0, 25)


# %%
ax2 = cg_Spatial1OnlyAcc.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only access score) + Environmental Prediction (all) Silhouette score")
ax2.set_xlim(0, 25)
ax2.axvline(5, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
c = ugg.COLORS
ax = cg_Spatial1OnlyAcc.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)

# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial1OnlyAcc.png", dpi=300, bbox_inches="tight")


# %%
c = ugg.COLORS
ax = cg_Spatial1OnlyAcc.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
ax.axvline(6, color=c[4], ymin=.02, ymax=.98)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
ax1 = cg_Spatial2OnlyConn.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (all) Silhouette score")
ax1.set_xlim(0, 25)


# %%
ax1 = cg_Spatial2OnlyConn.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (all) Silhouette score")
ax1.set_xlim(0, 25)
ax1.axvline(4, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
c = ugg.COLORS
ax = cg_Spatial2OnlyConn.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial2OnlyConn.png", dpi=300, bbox_inches="tight")


# %%
c = ugg.COLORS
ax = cg_Spatial2OnlyConn.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax.axvline(9, color=c[4], ymin=.02, ymax=.98)
ax.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
ax2 = cg_Spatial2OnlyAcc.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only access score) + Environmental Prediction (all) Silhouette score")
ax2.set_xlim(0, 25)


# %%
ax2 = cg_Spatial2OnlyAcc.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only access score) + Environmental Prediction (all) Silhouette score")
ax2.set_xlim(0, 25)
ax2.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(4, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
c = ugg.COLORS
ax = cg_Spatial2OnlyAcc.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial2OnlyAcc.png", dpi=300, bbox_inches="tight")

# %%
c = ugg.COLORS
ax = cg_Spatial2OnlyAcc.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax.axvline(9, color=c[4], ymin=.02, ymax=.98)
ax.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# ##### Env Pred Scores Cluster analysis

# %%
cg_OnlyConnEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_OnlyConnEnvScores.fit(dataTessOnlyConnEnvScores)

cg_OnlyAccEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_OnlyAccEnvScores.fit(dataTessOnlyAccEnvScores)

cg_Spatial1OnlyConnEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial1OnlyConnEnvScores.fit(dataTess1OnlyConnEnvScores)

cg_Spatial1OnlyAccEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial1OnlyAccEnvScores.fit(dataTess1OnlyAccEnvScores)

cg_Spatial2OnlyConnEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial2OnlyConnEnvScores.fit(dataTess2OnlyConnEnvScores)

cg_Spatial2OnlyAccEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial2OnlyAccEnvScores.fit(dataTess2OnlyAccEnvScores)


# %%
ax1 = cg_OnlyConnEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (All selected) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (env scores) Silhouette score")
ax1.set_xlim(0, 25)


# %%
ax1 = cg_OnlyConnEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (All selected) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (env scores) Silhouette score")
ax1.set_xlim(0, 25)
ax1.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(12, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
c = ugg.COLORS
ax = cg_OnlyConnEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatialAllEnvScoresOnlyConn.png", dpi=300, bbox_inches="tight")


# %%
c = ugg.COLORS
ax = cg_OnlyConnEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
ax.axvline(5, color=c[4], ymin=.02, ymax=.98)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# clear separation between components, even if not as clear as for env all, acc spatial all

# %%
ax2 = cg_OnlyAccEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (All selected) + Social Sustainability Indicators (only access score) + Environmental Prediction (env scores) Silhouette score")
ax2.set_xlim(0, 25)

# %%
ax2 = cg_OnlyAccEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (All selected) + Social Sustainability Indicators (only access score) + Environmental Prediction (env scores) Silhouette score")
ax2.set_xlim(0, 25)
ax2.axvline(10, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
c = ugg.COLORS
ax = cg_OnlyAccEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatialAllEnvScoresOnlyConn.png", dpi=300, bbox_inches="tight")


# %%
c = ugg.COLORS
ax = cg_OnlyAccEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax.axvline(8, color=c[4], ymin=.02, ymax=.98)
ax.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# Access score is also adequate separation of clusters

# %%
ax1 = cg_Spatial1OnlyConnEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (env scores) Silhouette score")
ax1.set_xlim(0, 25)


# %%
ax1 = cg_Spatial1OnlyConnEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (env scores) Silhouette score")
ax1.set_xlim(0, 25)
ax1.axvline(13, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
c = ugg.COLORS
ax = cg_Spatial1OnlyConnEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial1EnvScoresOnlyConn.png", dpi=300, bbox_inches="tight")


# %%
c = ugg.COLORS
ax = cg_Spatial1OnlyConnEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax.axvline(11, color=c[4], ymin=.02, ymax=.98)
ax.axvline(15, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# Silhouette scores for clusters of interest are high and the components are fairly differentiable.

# %%
ax2 = cg_Spatial1OnlyAccEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only access score) + Environmental Prediction (env scores) Silhouette score")
ax2.set_xlim(0, 25)

# %%
ax2 = cg_Spatial1OnlyAccEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only access score) + Environmental Prediction (env scores) Silhouette score")
ax2.set_xlim(0, 25)
ax2.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(11, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(10, color=c[4], ymin=.02, ymax=.98)

# %%
c = ugg.COLORS
ax = cg_Spatial1OnlyAccEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial1EnvScoresOnlyAcc.png", dpi=300, bbox_inches="tight")


# %%
c = ugg.COLORS
ax = cg_Spatial1OnlyAccEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
ax.axvline(6, color=c[4], ymin=.02, ymax=.98)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# Quite Stable component separation and silhouette scores reasonably high

# %%
ax1 = cg_Spatial2OnlyConnEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (env scores) Silhouette score")
ax1.set_xlim(0, 25)

# %%
ax1 = cg_Spatial2OnlyConnEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (env scores) Silhouette score")
ax1.set_xlim(0, 25)
ax1.axvline(10, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
c = ugg.COLORS
ax = cg_Spatial2OnlyConnEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial2EnvScoresOnlyConn.png", dpi=300, bbox_inches="tight")


# %%
c = ugg.COLORS
ax = cg_Spatial2OnlyConnEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax.axvline(11, color=c[4], ymin=.02, ymax=.98)
ax.axvline(12, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# Fairly stable component separation and high silhouette scores for the clusters of interest (7 and 11)

# %%
ax2 = cg_Spatial2OnlyAccEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only access score) + Environmental Prediction (env scores) Silhouette score")
ax2.set_xlim(0, 25)


# %%
ax2 = cg_Spatial2OnlyAccEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only access score) + Environmental Prediction (env scores) Silhouette score")
ax2.set_xlim(0, 25)
ax2.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(11, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(9, color=c[4], ymin=.02, ymax=.98)

# %%
c = ugg.COLORS
ax = cg_Spatial2OnlyAccEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial2EnvScoresOnlyAcc.png", dpi=300, bbox_inches="tight")

# %%
c = ugg.COLORS
ax = cg_Spatial2OnlyAccEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax.axvline(11, color=c[4], ymin=.02, ymax=.98)
ax.axvline(5, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# Again fairly decent component separation and silhouette scoring.

# %% [markdown]
# ##### Env Pred Character Cluster analysis

# %%
cg_SpatialAllOnlyConnEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_SpatialAllOnlyConnEnvChars.fit(dataTessOnlyConnEnvChars)

cg_SpatialAllOnlyAccEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_SpatialAllOnlyAccEnvChars.fit(dataTessOnlyAccEnvChars)

cg_Spatial1OnlyConnEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial1OnlyConnEnvChars.fit(dataTess1OnlyConnEnvChars)

cg_Spatial1OnlyAccEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial1OnlyAccEnvChars.fit(dataTess1OnlyAccEnvChars)

cg_Spatial2OnlyConnEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial2OnlyConnEnvChars.fit(dataTess2OnlyConnEnvChars)

cg_Spatial2OnlyAccEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial2OnlyAccEnvChars.fit(dataTess2OnlyAccEnvChars)



# %%
ax1 = cg_SpatialAllOnlyConnEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (All selected) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (characteristics) Silhouette score")
ax1.set_xlim(0, 25)


# %%
ax1 = cg_SpatialAllOnlyConnEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (All selected) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (characteristics) Silhouette score")
ax1.set_xlim(0, 25)
ax1.axvline(4, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
c = ugg.COLORS
ax = cg_SpatialAllOnlyConnEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatialAllEnvCharsOnlyConn.png", dpi=300, bbox_inches="tight")


# %%
c = ugg.COLORS
ax = cg_SpatialAllOnlyConnEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
ax.axvline(5, color=c[4], ymin=.02, ymax=.98)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# Not great silhouette scores but components are quite differentiable thus of interest.

# %%
ax2 = cg_SpatialAllOnlyAccEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (All selected) + Social Sustainability Indicators (only access score) + Environmental Prediction (characteristics) Silhouette score")
ax2.set_xlim(0, 25)
# ax.axvline(5, color=c[4], ymin=.02, ymax=.98)


# %%
ax2 = cg_SpatialAllOnlyAccEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (All selected) + Social Sustainability Indicators (only access score) + Environmental Prediction (characteristics) Silhouette score")
ax2.set_xlim(0, 25)
ax2.axvline(5, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
c = ugg.COLORS
ax = cg_SpatialAllOnlyAccEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatialAllEnvCharsOnlyAcc.png", dpi=300, bbox_inches="tight")


# %%
c = ugg.COLORS
ax = cg_SpatialAllOnlyAccEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax.axvline(8, color=c[4], ymin=.02, ymax=.98)
ax.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# Again not ideal but fairly reasonable component separation.

# %%
ax1 = cg_Spatial1OnlyConnEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (characteristics) Silhouette score")
ax1.set_xlim(0, 25)


# %%
ax1 = cg_Spatial1OnlyConnEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (characteristics) Silhouette score")
ax1.set_xlim(0, 25)
ax1.axvline(5, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
c = ugg.COLORS
ax = cg_Spatial1OnlyConnEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial1EnvCharsOnlyConn.png", dpi=300, bbox_inches="tight")


# %%
c = ugg.COLORS
ax = cg_Spatial1OnlyConnEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax.axvline(9, color=c[4], ymin=.02, ymax=.98)
ax.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# Silhouette scores too low and components to dense, thus not of interest.

# %%
ax2 = cg_Spatial1OnlyAccEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only access score) + Environmental Prediction (characteristics) Silhouette score")
ax2.set_xlim(0, 25)


# %%
ax2 = cg_Spatial1OnlyAccEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only access score) + Environmental Prediction (characteristics) Silhouette score")
ax2.set_xlim(0, 25)
ax2.axvline(5, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
c = ugg.COLORS
ax = cg_Spatial1OnlyAccEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial1EnvCharsOnlyAcc.png", dpi=300, bbox_inches="tight")


# %%
c = ugg.COLORS
ax = cg_Spatial1OnlyAccEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
ax.axvline(6, color=c[4], ymin=.02, ymax=.98)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
ax1 = cg_Spatial2OnlyConnEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (characteristics) Silhouette score")
ax1.set_xlim(0, 25)

# %%
ax1 = cg_Spatial2OnlyConnEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (characteristics) Silhouette score")
ax1.set_xlim(0, 25)
ax1.axvline(4, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(11, color=c[4], ymin=.02, ymax=.98)


# %%
c = ugg.COLORS
ax = cg_Spatial2OnlyConnEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial2EnvCharsOnlyConn.png", dpi=300, bbox_inches="tight")


# %%
c = ugg.COLORS
ax = cg_Spatial2OnlyConnEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
ax.axvline(5, color=c[4], ymin=.02, ymax=.98)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
ax2 = cg_Spatial2OnlyAccEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only access score) + Environmental Prediction (characteristics) Silhouette score")
ax2.set_xlim(0, 25)

# %%
ax2 = cg_Spatial2OnlyAccEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only access score) + Environmental Prediction (characteristics) Silhouette score")
ax2.set_xlim(0, 25)
ax2.axvline(4, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(11, color=c[4], ymin=.02, ymax=.98)


# %%
c = ugg.COLORS
ax = cg_Spatial2OnlyAccEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial2EnvCharsOnlyAcc.png", dpi=300, bbox_inches="tight")

# %%
c = ugg.COLORS
ax = cg_Spatial2OnlyAccEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_xlim(0, 25)
ax.axvline(6, color=c[4], ymin=.02, ymax=.98)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# #### Cluster Analysis Silhouette Scores Semi-Final

# %% [markdown]
# ##### Finalized set of silhouette scores Env Pred All

# %%
cg_Spatial1OnlyConn = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial1OnlyConn.fit(dataTessSpatial1OnlyConn)
ax1 = cg_Spatial1OnlyConn.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (all) Silhouette score")

cg_Spatial1OnlyAcc = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial1OnlyAcc.fit(dataTessSpatial1OnlyAcc)
ax2 = cg_Spatial1OnlyAcc.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only access score) + Environmental Prediction (all) Silhouette score")

# %%
ax1 = cg_Spatial1OnlyConn.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (all) Silhouette score")
ax1.axvline(5, color=c[4], ymin=.02, ymax=.98)


# %%
ax2 = cg_Spatial1OnlyAcc.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only access score) + Environmental Prediction (all) Silhouette score")
ax2.axvline(5, color=c[4], ymin=.02, ymax=.98)

# %%
cg_Spatial2OnlyConn = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial2OnlyConn.fit(dataTessSpatial2OnlyConn)
ax1 = cg_Spatial2OnlyConn.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (all) Silhouette score")

cg_Spatial2OnlyAcc = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial2OnlyAcc.fit(dataTessSpatial2OnlyAcc)
ax2 = cg_Spatial2OnlyAcc.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only access score) + Environmental Prediction (all) Silhouette score")

# %%
ax1 = cg_Spatial2OnlyConn.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (all) Silhouette score")
ax1.axvline(4, color=c[4], ymin=.02, ymax=.98)

# %%
ax2 = cg_Spatial2OnlyAcc.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only access score) + Environmental Prediction (all) Silhouette score")
ax2.axvline(4, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(6, color=c[4], ymin=.02, ymax=.98)

# %%
cg_Spatial3OnlyConn = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial3OnlyConn.fit(dataTessSpatial3OnlyConn)
ax1 = cg_Spatial3OnlyConn.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 3) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (all) Silhouette score")

cg_Spatial3OnlyAcc = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial3OnlyAcc.fit(dataTessSpatial3OnlyAcc)
ax2 = cg_Spatial3OnlyAcc.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 3) + Social Sustainability Indicators (only access score) + Environmental Prediction (all) Silhouette score")

# %%
ax1 = cg_Spatial3OnlyConn.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 3) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (all) Silhouette score")
ax1.axvline(5, color=c[4], ymin=.02, ymax=.98)


# %%
ax2 = cg_Spatial3OnlyAcc.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 3) + Social Sustainability Indicators (only access score) + Environmental Prediction (all) Silhouette score")
ax2.axvline(5, color=c[4], ymin=.02, ymax=.98)


# %%
cg_Spatial4OnlyConn = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial4OnlyConn.fit(dataTessSpatial4OnlyConn)
ax1 = cg_Spatial4OnlyConn.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 4) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (all) Silhouette score")

cg_Spatial4OnlyAcc = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial4OnlyAcc.fit(dataTessSpatial4OnlyAcc)
ax2 = cg_Spatial4OnlyAcc.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 4) + Social Sustainability Indicators (only access score) + Environmental Prediction (all) Silhouette score")

# %%
ax1 = cg_Spatial4OnlyConn.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 4) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (all) Silhouette score")
ax1.axvline(5, color=c[4], ymin=.02, ymax=.98)


# %%
ax2 = cg_Spatial4OnlyAcc.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 4) + Social Sustainability Indicators (only access score) + Environmental Prediction (all) Silhouette score")
ax2.axvline(5, color=c[4], ymin=.02, ymax=.98)


# %%
cg_Spatial5OnlyConn = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial5OnlyConn.fit(dataTessSpatial5OnlyConn)
ax1 = cg_Spatial5OnlyConn.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 5) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (all) Silhouette score")

cg_Spatial5OnlyAcc = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial5OnlyAcc.fit(dataTessSpatial5OnlyAcc)
ax2 = cg_Spatial5OnlyAcc.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 5) + Social Sustainability Indicators (only access score) + Environmental Prediction (all) Silhouette score")

# %%
ax1 = cg_Spatial5OnlyConn.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 5) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (all) Silhouette score")
ax1.axvline(4, color=c[4], ymin=.02, ymax=.98)


# %%
ax2 = cg_Spatial5OnlyAcc.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 5) + Social Sustainability Indicators (only access score) + Environmental Prediction (all) Silhouette score")
ax2.axvline(4, color=c[4], ymin=.02, ymax=.98)


# %% [markdown]
# ##### Finalized set of silhouette scores Env Pred Scores

# %%
cg_OnlyConnEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_OnlyConnEnvScores.fit(dataTessOnlyConnEnvScores)
ax1 = cg_OnlyConnEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only scores) Silhouette score")

cg_OnlyAccEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_OnlyAccEnvScores.fit(dataTessOnlyAccEnvScores)
ax2 = cg_OnlyAccEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form + Social Sustainability Indicators (only access scores) + Environmental Prediction (only scores) Silhouette score")

# %%
ax1 = cg_OnlyConnEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only scores) Silhouette score")
ax1.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(12, color=c[4], ymin=.02, ymax=.98)


# %%
ax2 = cg_OnlyAccEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form + Social Sustainability Indicators (only access scores) + Environmental Prediction (only scores) Silhouette score")
ax2.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(11, color=c[4], ymin=.02, ymax=.98)


# %%
cg_Spatial1OnlyConnEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial1OnlyConnEnvScores.fit(dataTess1OnlyConnEnvScores)
ax1 = cg_Spatial1OnlyConnEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only scores) Silhouette score")

cg_Spatial1OnlyAccEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial1OnlyAccEnvScores.fit(dataTess1OnlyAccEnvScores)
ax2 = cg_Spatial1OnlyAccEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only access score) + Environmental Prediction (only scores) Silhouette score")

# %%
ax1 = cg_Spatial1OnlyConnEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only scores) Silhouette score")
ax1.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(5, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(9, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(13, color=c[4], ymin=.02, ymax=.98)


# %%
ax2 = cg_Spatial1OnlyAccEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only access score) + Environmental Prediction (only scores) Silhouette score")
ax2.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(5, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(10, color=c[4], ymin=.02, ymax=.98)

# %%
cg_Spatial2OnlyConnEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial2OnlyConnEnvScores.fit(dataTess2OnlyConnEnvScores)
ax1 = cg_Spatial2OnlyConnEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only scores) Silhouette score")

cg_Spatial2OnlyAccEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial2OnlyAccEnvScores.fit(dataTess2OnlyAccEnvScores)
ax2 = cg_Spatial2OnlyAccEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only access score) + Environmental Prediction (only scores) Silhouette score")

# %%
ax1 = cg_Spatial2OnlyConnEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only scores) Silhouette score")
ax1.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(6, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(11, color=c[4], ymin=.02, ymax=.98)

# %%
ax2 = cg_Spatial2OnlyAccEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only access score) + Environmental Prediction (only scores) Silhouette score")
ax2.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(8, color=c[4], ymin=.02, ymax=.98)

# %%
cg_Spatial3OnlyConnEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial3OnlyConnEnvScores.fit(dataTess3OnlyConnEnvScores)
ax1 = cg_Spatial3OnlyConnEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 3) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only scores) Silhouette score")

cg_Spatial3OnlyAccEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial3OnlyAccEnvScores.fit(dataTess3OnlyAccEnvScores)
ax2 = cg_Spatial3OnlyAccEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 3) + Social Sustainability Indicators (only access score) + Environmental Prediction (only scores) Silhouette score")

# %%
ax1 = cg_Spatial3OnlyConnEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 3) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only scores) Silhouette score")
ax1.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(6, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(12, color=c[4], ymin=.02, ymax=.98)

# %%
ax2 = cg_Spatial3OnlyAccEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 3) + Social Sustainability Indicators (only access score) + Environmental Prediction (only scores) Silhouette score")
ax2.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(6, color=c[4], ymin=.02, ymax=.98)

# %%
cg_Spatial4OnlyConnEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial4OnlyConnEnvScores.fit(dataTess4OnlyConnEnvScores)
ax1 = cg_Spatial4OnlyConnEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 4) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only scores) Silhouette score")

cg_Spatial4OnlyAccEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial4OnlyAccEnvScores.fit(dataTess4OnlyAccEnvScores)
ax2 = cg_Spatial4OnlyAccEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 4) + Social Sustainability Indicators (only access score) + Environmental Prediction (only scores) Silhouette score")

# %%
cg_Spatial5OnlyConnEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial5OnlyConnEnvScores.fit(dataTess5OnlyConnEnvScores)
ax1 = cg_Spatial5OnlyConnEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 5) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only scores) Silhouette score")

cg_Spatial5OnlyAccEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial5OnlyAccEnvScores.fit(dataTess5OnlyAccEnvScores)
ax2 = cg_Spatial5OnlyAccEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 5) + Social Sustainability Indicators (only access score) + Environmental Prediction (only scores) Silhouette score")

# %% [markdown]
# ##### Finalized set of silhouette scores Env Pred Character

# %%
cg_SpatialAllOnlyConnEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_SpatialAllOnlyConnEnvChars.fit(dataTessOnlyConnEnvChars)
ax1 = cg_SpatialAllOnlyConnEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (All) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only env character) Silhouette score")

cg_SpatialAllOnlyAccEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_SpatialAllOnlyAccEnvChars.fit(dataTessOnlyAccEnvChars)
ax2 = cg_SpatialAllOnlyAccEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (All) + Social Sustainability Indicators (only access score) + Environmental Prediction (only env character) Silhouette score")

# %%
cg_Spatial1OnlyConnEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial1OnlyConnEnvChars.fit(dataTess1OnlyConnEnvChars)
ax1 = cg_Spatial1OnlyConnEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only env character) Silhouette score")

cg_Spatial1OnlyAccEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial1OnlyAccEnvChars.fit(dataTess1OnlyAccEnvChars)
ax2 = cg_Spatial1OnlyAccEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only access score) + Environmental Prediction (only env character) Silhouette score")

# %%
ax1 = cg_Spatial1OnlyConnEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only env character) Silhouette score")
ax1.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(5, color=c[4], ymin=.02, ymax=.98)

# %%
ax2 = cg_Spatial1OnlyAccEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 1) + Social Sustainability Indicators (only access score) + Environmental Prediction (only env character) Silhouette score")
ax2.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(5, color=c[4], ymin=.02, ymax=.98)

# %%
cg_Spatial2OnlyConnEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial2OnlyConnEnvChars.fit(dataTess2OnlyConnEnvChars)
ax1 = cg_Spatial2OnlyConnEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only env character) Silhouette score")

cg_Spatial2OnlyAccEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial2OnlyAccEnvChars.fit(dataTess2OnlyAccEnvChars)
ax2 = cg_Spatial2OnlyAccEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 2) + Social Sustainability Indicators (only access score) + Environmental Prediction (only env character) Silhouette score")

# %%
cg_Spatial3OnlyConnEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial3OnlyConnEnvChars.fit(dataTess3OnlyConnEnvChars)
ax1 = cg_Spatial3OnlyConnEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 3) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only env character) Silhouette score")

cg_Spatial3OnlyAccEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial3OnlyAccEnvChars.fit(dataTess3OnlyAccEnvChars)
ax2 = cg_Spatial3OnlyAccEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 3) + Social Sustainability Indicators (only access score) + Environmental Prediction (only env character) Silhouette score")

# %%
ax1 = cg_Spatial3OnlyConnEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 3) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only env character) Silhouette score")
ax1.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(5, color=c[4], ymin=.02, ymax=.98)

# %%
ax2 = cg_Spatial3OnlyAccEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 3) + Social Sustainability Indicators (only access score) + Environmental Prediction (only env character) Silhouette score")
ax2.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(5, color=c[4], ymin=.02, ymax=.98)

# %%
cg_Spatial4OnlyConnEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial4OnlyConnEnvChars.fit(dataTess4OnlyConnEnvChars)
ax1 = cg_Spatial4OnlyConnEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 4) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only env character) Silhouette score")

cg_Spatial4OnlyAccEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial4OnlyAccEnvChars.fit(dataTess4OnlyAccEnvChars)
ax2 = cg_Spatial4OnlyAccEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 4) + Social Sustainability Indicators (only access score) + Environmental Prediction (only env character) Silhouette score")

# %%
ax1 = cg_Spatial4OnlyConnEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 4) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only env character) Silhouette score")
ax1.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax1.axvline(5, color=c[4], ymin=.02, ymax=.98)

# %%
ax2 = cg_Spatial4OnlyAccEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 4) + Social Sustainability Indicators (only access score) + Environmental Prediction (only env character) Silhouette score")
ax2.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(5, color=c[4], ymin=.02, ymax=.98)

# %%
cg_Spatial5OnlyConnEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial5OnlyConnEnvChars.fit(dataTess5OnlyConnEnvChars)
ax1 = cg_Spatial5OnlyConnEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (combination 5) + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only env character) Silhouette score")

cg_Spatial5OnlyAccEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Spatial5OnlyAccEnvChars.fit(dataTess5OnlyAccEnvChars)
ax2 = cg_Spatial5OnlyAccEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 5) + Social Sustainability Indicators (only access score) + Environmental Prediction (only env character) Silhouette score")

# %%
ax2 = cg_Spatial5OnlyAccEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (combination 5) + Social Sustainability Indicators (only access score) + Environmental Prediction (only env character) Silhouette score")
ax2.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(4, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# ##### PCA Clustergrams

# %%
c = ugg.COLORS
ax = cg_Spatial1OnlyConn.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial1OnlyConn.png", dpi=300, bbox_inches="tight")

# %%
c = ugg.COLORS
ax = cg_Spatial1OnlyAcc.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial1OnlyAcc.png", dpi=300, bbox_inches="tight")

# %%
c = ugg.COLORS
ax = cg_Spatial2OnlyConn.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial2OnlyConn.png", dpi=300, bbox_inches="tight")


# %%

c = ugg.COLORS
ax = cg_Spatial2OnlyAcc.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial2OnlyAcc.png", dpi=300, bbox_inches="tight")

# %%
c = ugg.COLORS
ax = cg_Spatial3OnlyConn.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial3OnlyConn.png", dpi=300, bbox_inches="tight")


# %%

c = ugg.COLORS
ax = cg_Spatial3OnlyAcc.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial3OnlyAcc.png", dpi=300, bbox_inches="tight")

# %%
c = ugg.COLORS
ax = cg_Spatial4OnlyConn.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial4OnlyConn.png", dpi=300, bbox_inches="tight")


# %%

c = ugg.COLORS
ax = cg_Spatial4OnlyAcc.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial4OnlyAcc.png", dpi=300, bbox_inches="tight")

# %%
c = ugg.COLORS
ax = cg_Spatial5OnlyConn.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial5OnlyConn.png", dpi=300, bbox_inches="tight")


# %%

c = ugg.COLORS
ax = cg_Spatial5OnlyAcc.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial5OnlyAcc.png", dpi=300, bbox_inches="tight")

# %%
c = ugg.COLORS
ax = cg_OnlyConnEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramOnlyConnEnvScores.png", dpi=300, bbox_inches="tight")


# %%

c = ugg.COLORS
ax = cg_OnlyAccEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramOnlyAccEnvScores.png", dpi=300, bbox_inches="tight")

# %%
c = ugg.COLORS
ax = cg_Spatial1OnlyConnEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial1OnlyConnEnvScores.png", dpi=300, bbox_inches="tight")


# %%

c = ugg.COLORS
ax = cg_Spatial1OnlyAccEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial1OnlyAccEnvScores.png", dpi=300, bbox_inches="tight")

# %%
c = ugg.COLORS
ax = cg_Spatial2OnlyConnEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial2OnlyConnEnvScores.png", dpi=300, bbox_inches="tight")


# %%

c = ugg.COLORS
ax = cg_Spatial2OnlyAccEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial2OnlyAccEnvScores.png", dpi=300, bbox_inches="tight")

# %%
c = ugg.COLORS
ax = cg_Spatial3OnlyConnEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_ylim(-3, 10)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial3OnlyConnEnvScores.png", dpi=300, bbox_inches="tight")


# %%

c = ugg.COLORS
ax = cg_Spatial3OnlyAccEnvScores.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
ax.set_ylim(-3, 10)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial3OnlyAccEnvScores.png", dpi=300, bbox_inches="tight")

# %%
c = ugg.COLORS
ax = cg_Spatial1OnlyConnEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial1OnlyConnEnvChars.png", dpi=300, bbox_inches="tight")


# %%

c = ugg.COLORS
ax = cg_Spatial1OnlyAccEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial1OnlyAccEnvChars.png", dpi=300, bbox_inches="tight")

# %%
c = ugg.COLORS
ax = cg_Spatial3OnlyConnEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial3OnlyConnEnvChars.png", dpi=300, bbox_inches="tight")


# %%

c = ugg.COLORS
ax = cg_Spatial3OnlyAccEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial3OnlyAccEnvChars.png", dpi=300, bbox_inches="tight")

# %%
c = ugg.COLORS
ax = cg_Spatial4OnlyConnEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial4OnlyConnEnvChars.png", dpi=300, bbox_inches="tight")


# %%

c = ugg.COLORS
ax = cg_Spatial4OnlyAccEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial4OnlyAccEnvChars.png", dpi=300, bbox_inches="tight")

# %%

c = ugg.COLORS
ax = cg_Spatial5OnlyAccEnvChars.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramSpatial5OnlyAccEnvChars.png", dpi=300, bbox_inches="tight")

# %%
with open(folder + 'clustergram_Spatial1OnlyConn.pickle','wb') as f:
    pickle.dump(cg_Spatial1OnlyConn, f)
with open(folder + 'clustergram_Spatial1OnlyAcc.pickle','wb') as f:
    pickle.dump(cg_Spatial1OnlyAcc, f)
with open(folder + 'clustergram_Spatial2OnlyConn.pickle','wb') as f:
    pickle.dump(cg_Spatial2OnlyConn, f)
with open(folder + 'clustergram_Spatial2OnlyAcc.pickle','wb') as f:
    pickle.dump(cg_Spatial2OnlyAcc, f)
with open(folder + 'clustergram_Spatial3OnlyConn.pickle','wb') as f:
    pickle.dump(cg_Spatial3OnlyConn, f)
with open(folder + 'clustergram_Spatial3OnlyAcc.pickle','wb') as f:
    pickle.dump(cg_Spatial3OnlyAcc, f)
with open(folder + 'clustergram_Spatial4OnlyConn.pickle','wb') as f:
    pickle.dump(cg_Spatial4OnlyConn, f)
with open(folder + 'clustergram_Spatial4OnlyAcc.pickle','wb') as f:
    pickle.dump(cg_Spatial4OnlyAcc, f)
with open(folder + 'clustergram_Spatial5OnlyConn.pickle','wb') as f:
    pickle.dump(cg_Spatial5OnlyConn, f)
with open(folder + 'clustergram_Spatial5OnlyAcc.pickle','wb') as f:
    pickle.dump(cg_Spatial5OnlyAcc, f)

# %%
with open(folder + 'clustergram_OnlyConnEnvScores.pickle','wb') as f:
    pickle.dump(cg_OnlyConnEnvScores, f)
with open(folder + 'clustergram_OnlyAccEnvScores.pickle','wb') as f:
    pickle.dump(cg_OnlyAccEnvScores, f)

with open(folder + 'clustergram_Spatial1OnlyConnEnvScores.pickle','wb') as f:
    pickle.dump(cg_Spatial1OnlyConnEnvScores, f)
with open(folder + 'clustergram_Spatial1OnlyAccEnvScores.pickle','wb') as f:
    pickle.dump(cg_Spatial1OnlyAccEnvScores, f)

with open(folder + 'clustergram_Spatial2OnlyConnEnvScores.pickle','wb') as f:
    pickle.dump(cg_Spatial2OnlyConnEnvScores, f)
with open(folder + 'clustergram_Spatial2OnlyAccEnvScores.pickle','wb') as f:
    pickle.dump(cg_Spatial2OnlyAccEnvScores, f)

with open(folder + 'clustergram_Spatial3OnlyConnEnvScores.pickle','wb') as f:
    pickle.dump(cg_Spatial3OnlyConnEnvScores, f)
with open(folder + 'clustergram_Spatial3OnlyAccEnvScores.pickle','wb') as f:
    pickle.dump(cg_Spatial3OnlyAccEnvScores, f)

# %%
with open(folder + 'clustergram_Spatial1OnlyConnEnvChars.pickle','wb') as f:
    pickle.dump(cg_Spatial1OnlyConnEnvChars, f)
with open(folder + 'clustergram_Spatial1OnlyAccEnvChars.pickle','wb') as f:
    pickle.dump(cg_Spatial1OnlyAccEnvChars, f)

with open(folder + 'clustergram_Spatial3OnlyConnEnvChars.pickle','wb') as f:
    pickle.dump(cg_Spatial3OnlyConnEnvChars, f)
with open(folder + 'clustergram_Spatial3OnlyAccEnvChars.pickle','wb') as f:
    pickle.dump(cg_Spatial3OnlyAccEnvChars, f)

with open(folder + 'clustergram_Spatial4OnlyConnEnvChars.pickle','wb') as f:
    pickle.dump(cg_Spatial4OnlyConnEnvChars, f)
with open(folder + 'clustergram_Spatial4OnlyAccEnvChars.pickle','wb') as f:
    pickle.dump(cg_Spatial4OnlyAccEnvChars, f)


with open(folder + 'clustergram_Spatial5OnlyAccEnvChars.pickle','wb') as f:
    pickle.dump(cg_Spatial5OnlyAccEnvChars, f)

# %% [markdown]
# #### Second to last set of silhouette scores

# %% [markdown]
# ##### All Env Pred

# %%
cg_Full = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Full.fit(dataTessAll)
ax1 = cg_Full.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (all) + Environmental Prediction (all) Silhouette score")

# %% [markdown]
# Above is not worth clustering - dimensionality too high

# %%
cg_FullEnvNoPop = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_FullEnvNoPop.fit(dataTessNoPop)
ax1 = cg_FullEnvNoPop.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (No Population or Housing data) + Environmental Prediction (all) Silhouette score")

# %% [markdown]
# Also too high dimensionality to be worthwhile

# %%
cg_FullEnvNoSocScore = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_FullEnvNoSocScore.fit(dataTessNoSocScore)
ax1 = cg_FullEnvNoSocScore.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (no access score) + Environmental Prediction (all) Silhouette score")

# %% [markdown]
# Also does not make differentiable clusters

# %%
cg_FullEnvNoSocScoreNoH = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_FullEnvNoSocScoreNoH.fit(dataTessNoSocScoreNoH)
ax1 = cg_FullEnvNoSocScoreNoH.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (no access score or housing data) + Environmental Prediction (all) Silhouette score")

# %% [markdown]
# Does not give differentiable enough clusters

# %%
cg_FullEnvOnlyConn = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_FullEnvOnlyConn.fit(dataTessOnlyConn)
ax1 = cg_FullEnvOnlyConn.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (only connectivity data) + Environmental Prediction (all) Silhouette score")

# %% [markdown]
# ##### Env Scoring

# %%
cg_FullSocEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_FullSocEnvScores.fit(dataTessEnvScores)
ax1 = cg_FullSocEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (all) + Environmental Prediction (only scores) Silhouette score")

# %%
cg_NoPopEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_NoPopEnvScores.fit(dataTessNoPopEnvScores)
ax1 = cg_NoPopEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (no population or housing data) + Environmental Prediction (only scores) Silhouette score")

# %%
cg_NoSocScoreEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_NoSocScoreEnvScores.fit(dataTessNoSocScoreEnvScores)
ax1 = cg_NoSocScoreEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (no access score) + Environmental Prediction (only scores) Silhouette score")

# %%
cg_NoSocScoreNoHEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_NoSocScoreNoHEnvScores.fit(dataTessNoSocScoreNoHEnvScores)
ax1 = cg_NoSocScoreNoHEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (No access score or housing data) + Environmental Prediction (only scores) Silhouette score")

# %%
cg_OnlyConnEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_OnlyConnEnvScores.fit(dataTessOnlyConnEnvScores)
ax1 = cg_OnlyConnEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only scores) Silhouette score")

# %% [markdown]
# Only one that gives somewhat differentiable clusters

# %%
cg_OnlyAccEnvScores = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_OnlyAccEnvScores.fit(dataTessOnlyAccEnvScores)
ax1 = cg_OnlyAccEnvScores.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (only access scores) + Environmental Prediction (only scores) Silhouette score")

# %% [markdown]
# ##### Environmental Prediction Character

# %%
cg_FullSocEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_FullSocEnvChars.fit(dataTessEnvChars)
ax1 = cg_FullSocEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (all) + Environmental Prediction (only character) Silhouette score")

# %%
cg_NoPopEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_NoPopEnvChars.fit(dataTessNoPopEnvChars)
ax1 = cg_NoPopEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (no population or housing data) + Environmental Prediction (only character) Silhouette score")

# %%
cg_NoSocScoreEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_NoSocScoreEnvChars.fit(dataTessNoSocScoreEnvChars)
ax1 = cg_NoSocScoreEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (no access score) + Environmental Prediction (only character) Silhouette score")

# %%
cg_NoSocScoreNoHEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_NoSocScoreNoHEnvChars.fit(dataTessNoSocScoreNoHEnvChars)
ax1 = cg_NoSocScoreNoHEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (No access score or housing data) + Environmental Prediction (only character) Silhouette score")

# %%
cg_OnlyConnEnvChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_OnlyConnEnvChars.fit(dataTessOnlyConnEnvChars)
ax1 = cg_OnlyConnEnvChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (only connectivity indicators) + Environmental Prediction (only character) Silhouette score")

# %%
cg_OnlyConnAccChars = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_OnlyConnAccChars.fit(dataTessOnlyAccEnvChars)
ax1 = cg_OnlyConnAccChars.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form + Social Sustainability Indicators (only access scores) + Environmental Prediction (only character) Silhouette score")

# %% [markdown]
# Just not good enough for follow-up analysis -> however with reduction in dimensionality might be better

# %% [markdown]
# #### Clustering

# %% [markdown]
# ##### Clustering Final set

# %%
# Env All
km_EnvAll_OnlyConn_SpatialAll_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatialAllOnlyConn)
km_EnvAll_OnlyConn_SpatialAll_5  = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatialAllOnlyConn)
km_EnvAll_OnlyConn_SpatialAll_7  = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatialAllOnlyConn)
km_EnvAll_OnlyConn_SpatialAll_11  = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatialAllOnlyConn)

km_EnvAll_OnlyAcc_SpatialAll_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatialAllOnlyAcc)
km_EnvAll_OnlyAcc_SpatialAll_5  = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatialAllOnlyAcc)
km_EnvAll_OnlyAcc_SpatialAll_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatialAllOnlyAcc)
km_EnvAll_OnlyAcc_SpatialAll_11  = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatialAllOnlyAcc)

# Env Scores
km_EnvScores_OnlyConn_SpatialAll_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyConnEnvScores)
km_EnvScores_OnlyConn_SpatialAll_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyConnEnvScores)
km_EnvScores_OnlyConn_SpatialAll_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyConnEnvScores)
km_EnvScores_OnlyConn_SpatialAll_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyConnEnvScores)

km_EnvScores_OnlyAcc_SpatialAll_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyAccEnvScores)
km_EnvScores_OnlyAcc_SpatialAll_8 = MiniBatchKMeans(n_clusters=8, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyAccEnvScores)
km_EnvScores_OnlyAcc_SpatialAll_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyAccEnvScores)
km_EnvScores_OnlyAcc_SpatialAll_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyAccEnvScores)

km_EnvScores_OnlyConn_Spatial1_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyConnEnvScores)
km_EnvScores_OnlyConn_Spatial1_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyConnEnvScores)
km_EnvScores_OnlyConn_Spatial1_13 = MiniBatchKMeans(n_clusters=13, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyConnEnvScores)
km_EnvScores_OnlyConn_Spatial1_15 = MiniBatchKMeans(n_clusters=15, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyConnEnvScores)

km_EnvScores_OnlyAcc_Spatial1_6 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyAccEnvScores)
km_EnvScores_OnlyAcc_Spatial1_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyAccEnvScores)
km_EnvScores_OnlyAcc_Spatial1_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyAccEnvScores)
km_EnvScores_OnlyAcc_Spatial1_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyAccEnvScores)

km_EnvScores_OnlyConn_Spatial2_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2OnlyConnEnvScores)
km_EnvScores_OnlyConn_Spatial2_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2OnlyConnEnvScores)
km_EnvScores_OnlyConn_Spatial2_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2OnlyConnEnvScores)
km_EnvScores_OnlyConn_Spatial2_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2OnlyConnEnvScores)

km_EnvScores_OnlyAcc_Spatial2_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2OnlyAccEnvScores)
km_EnvScores_OnlyAcc_Spatial2_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2OnlyAccEnvScores)
km_EnvScores_OnlyAcc_Spatial2_9 = MiniBatchKMeans(n_clusters=9, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2OnlyAccEnvScores)
km_EnvScores_OnlyAcc_Spatial2_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2OnlyAccEnvScores)


# Env character
km_EnvChars_OnlyConn_SpatialAll_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyConnEnvChars)
km_EnvChars_OnlyConn_SpatialAll_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyConnEnvChars)
km_EnvChars_OnlyConn_SpatialAll_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyConnEnvChars)
km_EnvChars_OnlyConn_SpatialAll_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyConnEnvChars)

km_EnvChars_OnlyAcc_SpatialAll_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyAccEnvChars)
km_EnvChars_OnlyAcc_SpatialAll_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyAccEnvChars)
km_EnvChars_OnlyAcc_SpatialAll_8 = MiniBatchKMeans(n_clusters=8, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyAccEnvChars)
km_EnvChars_OnlyAcc_SpatialAll_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyAccEnvChars)


# %% [markdown]
# Saving final clustergrams

# %%
with open(folder + 'clustergram_SpatialAllEnvAllOnlyConn.pickle','wb') as f:
    pickle.dump(cg_SpatialAllOnlyConn, f)
with open(folder + 'clustergram_SpatialAllEnvAllOnlyAcc.pickle','wb') as f:
    pickle.dump(cg_SpatialAllOnlyAcc, f)
with open(folder + 'clustergram_Spatial1EnvAllOnlyConn.pickle','wb') as f:
    pickle.dump(cg_Spatial1OnlyConn, f)
with open(folder + 'clustergram_Spatial1EnvAllOnlyAcc.pickle','wb') as f:
    pickle.dump(cg_Spatial1OnlyAcc, f)
with open(folder + 'clustergram_Spatial2EnvAllOnlyConn.pickle','wb') as f:
    pickle.dump(cg_Spatial2OnlyConn, f)
with open(folder + 'clustergram_Spatial2EnvAllOnlyAcc.pickle','wb') as f:
    pickle.dump(cg_Spatial2OnlyAcc, f)

with open(folder + 'clustergram_SpatialAllEnvScoresOnlyConn.pickle','wb') as f:
    pickle.dump(cg_OnlyConnEnvScores, f)
with open(folder + 'clustergram_SpatialAllEnvScoresOnlyAcc.pickle','wb') as f:
    pickle.dump(cg_OnlyAccEnvScores, f)
with open(folder + 'clustergram_Spatial1EnvScoresOnlyConn.pickle','wb') as f:
    pickle.dump(cg_Spatial1OnlyConnEnvScores, f)
with open(folder + 'clustergram_Spatial1EnvScoresOnlyAcc.pickle','wb') as f:
    pickle.dump(cg_Spatial1OnlyAccEnvScores, f)
with open(folder + 'clustergram_Spatial2EnvScoresOnlyConn.pickle','wb') as f:
    pickle.dump(cg_Spatial2OnlyConnEnvScores, f)
with open(folder + 'clustergram_Spatial2EnvScoresOnlyAcc.pickle','wb') as f:
    pickle.dump(cg_Spatial2OnlyAccEnvScores, f)

with open(folder + 'clustergram_SpatialAllEnvCharsOnlyConn.pickle','wb') as f:
    pickle.dump(cg_SpatialAllOnlyConnEnvChars, f)
with open(folder + 'clustergram_SpatialAllEnvCharsOnlyAcc.pickle','wb') as f:
    pickle.dump(cg_SpatialAllOnlyAccEnvChars, f)
with open(folder + 'clustergram_Spatial1EnvCharsOnlyConn.pickle','wb') as f:
    pickle.dump(cg_Spatial1OnlyConnEnvChars, f)
with open(folder + 'clustergram_Spatial1EnvCharsOnlyAcc.pickle','wb') as f:
    pickle.dump(cg_Spatial1OnlyAccEnvChars, f)
with open(folder + 'clustergram_Spatial2EnvCharsOnlyConn.pickle','wb') as f:
    pickle.dump(cg_Spatial2OnlyConnEnvChars, f)
with open(folder + 'clustergram_Spatial2EnvCharsOnlyAcc.pickle','wb') as f:
    pickle.dump(cg_Spatial2OnlyAccEnvChars, f)

# %% [markdown]
# Saving Final KMeans Clusters

# %%
# Env All
with open(folder + 'km_EnvAll_OnlyConn_SpatialAll_4.pickle','wb') as f:
    pickle.dump(km_EnvAll_OnlyConn_SpatialAll_4, f)
with open(folder + 'km_EnvAll_OnlyConn_SpatialAll_5.pickle','wb') as f:
    pickle.dump(km_EnvAll_OnlyConn_SpatialAll_5, f)
with open(folder + 'km_EnvAll_OnlyConn_SpatialAll_7.pickle','wb') as f:
    pickle.dump(km_EnvAll_OnlyConn_SpatialAll_7, f)
with open(folder + 'km_EnvAll_OnlyConn_SpatialAll_11.pickle','wb') as f:
    pickle.dump(km_EnvAll_OnlyConn_SpatialAll_11, f)

with open(folder + 'km_EnvAll_OnlyAcc_SpatialAll_4.pickle','wb') as f:
    pickle.dump(km_EnvAll_OnlyAcc_SpatialAll_4, f)
with open(folder + 'km_EnvAll_OnlyAcc_SpatialAll_5.pickle','wb') as f:
    pickle.dump(km_EnvAll_OnlyAcc_SpatialAll_5, f)
with open(folder + 'km_EnvAll_OnlyAcc_SpatialAll_7.pickle','wb') as f:
    pickle.dump(km_EnvAll_OnlyAcc_SpatialAll_7, f)
with open(folder + 'km_EnvAll_OnlyAcc_SpatialAll_11.pickle','wb') as f:
    pickle.dump(km_EnvAll_OnlyAcc_SpatialAll_11, f)

# Env Score
with open(folder + 'km_EnvScores_OnlyConn_SpatialAll_5.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyConn_SpatialAll_5, f)
with open(folder + 'km_EnvScores_OnlyConn_SpatialAll_7.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyConn_SpatialAll_7, f)
with open(folder + 'km_EnvScores_OnlyConn_SpatialAll_11.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyConn_SpatialAll_11, f)
with open(folder + 'km_EnvScores_OnlyConn_SpatialAll_12.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyConn_SpatialAll_12, f)

with open(folder + 'km_EnvScores_OnlyAcc_SpatialAll_7.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyAcc_SpatialAll_7, f)
with open(folder + 'km_EnvScores_OnlyAcc_SpatialAll_8.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyAcc_SpatialAll_8, f)
with open(folder + 'km_EnvScores_OnlyAcc_SpatialAll_10.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyAcc_SpatialAll_10, f)
with open(folder + 'km_EnvScores_OnlyAcc_SpatialAll_11.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyAcc_SpatialAll_11, f)

with open(folder + 'km_EnvScores_OnlyConn_Spatial1_7.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyConn_Spatial1_7, f)
with open(folder + 'km_EnvScores_OnlyConn_Spatial1_11.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyConn_Spatial1_11, f)
with open(folder + 'km_EnvScores_OnlyConn_Spatial1_13.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyConn_Spatial1_13, f)
with open(folder + 'km_EnvScores_OnlyConn_Spatial1_15.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyConn_Spatial1_15, f)

with open(folder + 'km_EnvScores_OnlyAcc_Spatial1_6.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyAcc_Spatial1_6, f)
with open(folder + 'km_EnvScores_OnlyAcc_Spatial1_7.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyAcc_Spatial1_7, f)
with open(folder + 'km_EnvScores_OnlyAcc_Spatial1_10.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyAcc_Spatial1_10, f)
with open(folder + 'km_EnvScores_OnlyAcc_Spatial1_11.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyAcc_Spatial1_11, f)

with open(folder + 'km_EnvScores_OnlyConn_Spatial2_7.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyConn_Spatial2_7, f)
with open(folder + 'km_EnvScores_OnlyConn_Spatial2_10.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyConn_Spatial2_10, f)
with open(folder + 'km_EnvScores_OnlyConn_Spatial2_11.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyConn_Spatial2_11, f)
with open(folder + 'km_EnvScores_OnlyConn_Spatial2_12.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyConn_Spatial2_12, f)

with open(folder + 'km_EnvScores_OnlyAcc_Spatial2_5.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyAcc_Spatial2_5, f)
with open(folder + 'km_EnvScores_OnlyAcc_Spatial2_7.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyAcc_Spatial2_7, f)
with open(folder + 'km_EnvScores_OnlyAcc_Spatial2_9.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyAcc_Spatial2_9, f)
with open(folder + 'km_EnvScores_OnlyAcc_Spatial2_11.pickle','wb') as f:
    pickle.dump(km_EnvScores_OnlyAcc_Spatial2_11, f)

# Env Character
with open(folder + 'km_EnvChars_OnlyConn_SpatialAll_4.pickle','wb') as f:
    pickle.dump(km_EnvChars_OnlyConn_SpatialAll_4, f)
with open(folder + 'km_EnvChars_OnlyConn_SpatialAll_5.pickle','wb') as f:
    pickle.dump(km_EnvChars_OnlyConn_SpatialAll_5, f)
with open(folder + 'km_EnvChars_OnlyConn_SpatialAll_7.pickle','wb') as f:
    pickle.dump(km_EnvChars_OnlyConn_SpatialAll_7, f)
with open(folder + 'km_EnvChars_OnlyConn_SpatialAll_11.pickle','wb') as f:
    pickle.dump(km_EnvChars_OnlyConn_SpatialAll_11, f)

with open(folder + 'km_EnvChars_OnlyAcc_SpatialAll_5.pickle','wb') as f:
    pickle.dump(km_EnvChars_OnlyAcc_SpatialAll_5, f)
with open(folder + 'km_EnvChars_OnlyAcc_SpatialAll_7.pickle','wb') as f:
    pickle.dump(km_EnvChars_OnlyAcc_SpatialAll_7, f)
with open(folder + 'km_EnvChars_OnlyAcc_SpatialAll_8.pickle','wb') as f:
    pickle.dump(km_EnvChars_OnlyAcc_SpatialAll_8, f)
with open(folder + 'km_EnvChars_OnlyAcc_SpatialAll_11.pickle','wb') as f:
    pickle.dump(km_EnvChars_OnlyAcc_SpatialAll_11, f)

# %% [markdown]
# ##### Clustering semi-final set

# %%
km_EnvAll_OnlyConn_Spatial1_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial1OnlyConn)
km_EnvAll_OnlyConn_Spatial1_6  = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial1OnlyConn)
km_EnvAll_OnlyConn_Spatial1_9  = MiniBatchKMeans(n_clusters=9, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial1OnlyConn)
km_EnvAll_OnlyConn_Spatial1_10  = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial1OnlyConn)

km_EnvAll_OnlyAcc_Spatial1_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial1OnlyAcc)
km_EnvAll_OnlyAcc_Spatial1_6  = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial1OnlyAcc)

# %%
km_EnvAll_OnlyConn_Spatial2_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial2OnlyConn)

km_EnvAll_OnlyAcc_Spatial2_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial2OnlyAcc)
km_EnvAll_OnlyAcc_Spatial2_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial2OnlyAcc)
km_EnvAll_OnlyAcc_Spatial2_6  = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial2OnlyAcc)
km_EnvAll_OnlyAcc_Spatial2_8 = MiniBatchKMeans(n_clusters=8, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial2OnlyAcc)
km_EnvAll_OnlyAcc_Spatial2_13 = MiniBatchKMeans(n_clusters=13, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial2OnlyAcc)


# %%
km_EnvAll_OnlyConn_Spatial3_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial3OnlyConn)
km_EnvAll_OnlyConn_Spatial3_6 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial3OnlyConn)

km_EnvAll_OnlyAcc_Spatial3_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial3OnlyAcc)
km_EnvAll_OnlyAcc_Spatial3_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial3OnlyAcc)
km_EnvAll_OnlyAcc_Spatial3_6 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial3OnlyAcc)
km_EnvAll_OnlyAcc_Spatial3_9 = MiniBatchKMeans(n_clusters=9, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial3OnlyAcc)

# %%
km_EnvAll_OnlyConn_Spatial4_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial4OnlyConn)
km_EnvAll_OnlyConn_Spatial4_6 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial4OnlyConn)
km_EnvAll_OnlyConn_Spatial4_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial4OnlyConn)

km_EnvAll_OnlyAcc_Spatial4_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial4OnlyAcc)
km_EnvAll_OnlyAcc_Spatial4_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial4OnlyAcc)
km_EnvAll_OnlyAcc_Spatial4_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial4OnlyAcc)

# %%
km_EnvAll_OnlyConn_Spatial5_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial5OnlyConn)
km_EnvAll_OnlyConn_Spatial5_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial5OnlyConn)
km_EnvAll_OnlyConn_Spatial5_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial5OnlyConn)

km_EnvAll_OnlyAcc_Spatial5_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial5OnlyAcc)
km_EnvAll_OnlyAcc_Spatial5_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial5OnlyAcc)
km_EnvAll_OnlyAcc_Spatial5_15 = MiniBatchKMeans(n_clusters=15, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessSpatial5OnlyAcc)

# %%
km_EnvScores_OnlyConn_SpatialAll_3 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyConnEnvScores)
km_EnvScores_OnlyConn_SpatialAll_6 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyConnEnvScores)
km_EnvScores_OnlyConn_SpatialAll_9 = MiniBatchKMeans(n_clusters=9, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyConnEnvScores)
km_EnvScores_OnlyConn_SpatialAll_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyConnEnvScores)

km_EnvScores_OnlyAcc_SpatialAll_3 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyAccEnvScores)
km_EnvScores_OnlyAcc_SpatialAll_8 = MiniBatchKMeans(n_clusters=8, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyAccEnvScores)
km_EnvScores_OnlyAcc_SpatialAll_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTessOnlyAccEnvScores)


# %%
km_EnvScores_OnlyConn_Spatial1_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyConnEnvScores)
km_EnvScores_OnlyConn_Spatial1_8 = MiniBatchKMeans(n_clusters=8, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyConnEnvScores)
km_EnvScores_OnlyConn_Spatial1_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyConnEnvScores)
km_EnvScores_OnlyConn_Spatial1_13 = MiniBatchKMeans(n_clusters=13, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyConnEnvScores)

km_EnvScores_OnlyAcc_Spatial1_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyAccEnvScores)
km_EnvScores_OnlyAcc_Spatial1_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyAccEnvScores)
km_EnvScores_OnlyAcc_Spatial1_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyAccEnvScores)
km_EnvScores_OnlyAcc_Spatial1_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyAccEnvScores)


# %%
km_EnvScores_OnlyConn_Spatial2_3 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2OnlyConnEnvScores)
km_EnvScores_OnlyConn_Spatial2_6 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2OnlyConnEnvScores)
km_EnvScores_OnlyConn_Spatial2_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2OnlyConnEnvScores)
km_EnvScores_OnlyConn_Spatial2_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2OnlyConnEnvScores)

km_EnvScores_OnlyAcc_Spatial2_3 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2OnlyAccEnvScores)
km_EnvScores_OnlyAcc_Spatial2_6 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2OnlyAccEnvScores)
km_EnvScores_OnlyAcc_Spatial2_8 = MiniBatchKMeans(n_clusters=8, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2OnlyAccEnvScores)
km_EnvScores_OnlyAcc_Spatial2_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2OnlyAccEnvScores)


# %%
km_EnvScores_OnlyConn_Spatial3_3 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3OnlyConnEnvScores)
km_EnvScores_OnlyConn_Spatial3_6 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3OnlyConnEnvScores)
km_EnvScores_OnlyConn_Spatial3_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3OnlyConnEnvScores)
km_EnvScores_OnlyConn_Spatial3_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3OnlyConnEnvScores)

km_EnvScores_OnlyAcc_Spatial3_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3OnlyAccEnvScores)
km_EnvScores_OnlyAcc_Spatial3_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3OnlyAccEnvScores)
km_EnvScores_OnlyAcc_Spatial3_6 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3OnlyAccEnvScores)
km_EnvScores_OnlyAcc_Spatial3_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3OnlyAccEnvScores)


# %%
km_EnvChars_OnlyConn_Spatial1_3 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyConnEnvChars)
km_EnvChars_OnlyConn_Spatial1_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyConnEnvChars)
km_EnvChars_OnlyConn_Spatial1_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyConnEnvChars)
km_EnvChars_OnlyConn_Spatial1_6 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyConnEnvChars)

km_EnvChars_OnlyAcc_Spatial1_3 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyAccEnvChars)
km_EnvChars_OnlyAcc_Spatial1_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyAccEnvChars)
km_EnvChars_OnlyAcc_Spatial1_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyAccEnvChars)
km_EnvChars_OnlyAcc_Spatial1_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1OnlyAccEnvChars)


# %%
km_EnvChars_OnlyConn_Spatial3_3 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3OnlyConnEnvChars)
km_EnvChars_OnlyConn_Spatial3_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3OnlyConnEnvChars)
km_EnvChars_OnlyConn_Spatial3_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3OnlyConnEnvChars)
km_EnvChars_OnlyConn_Spatial3_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3OnlyConnEnvChars)

km_EnvChars_OnlyAcc_Spatial3_3 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3OnlyAccEnvChars)
km_EnvChars_OnlyAcc_Spatial3_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3OnlyAccEnvChars)
km_EnvChars_OnlyAcc_Spatial3_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3OnlyAccEnvChars)
km_EnvChars_OnlyAcc_Spatial3_6 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3OnlyAccEnvChars)
km_EnvChars_OnlyAcc_Spatial3_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3OnlyAccEnvChars)
km_EnvChars_OnlyAcc_Spatial3_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3OnlyAccEnvChars)


# %%
km_EnvChars_OnlyConn_Spatial4_3 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4OnlyConnEnvChars)
km_EnvChars_OnlyConn_Spatial4_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4OnlyConnEnvChars)
km_EnvChars_OnlyConn_Spatial4_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4OnlyConnEnvChars)
km_EnvChars_OnlyConn_Spatial4_6 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4OnlyConnEnvChars)
km_EnvChars_OnlyConn_Spatial4_8 = MiniBatchKMeans(n_clusters=8, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4OnlyConnEnvChars)
km_EnvChars_OnlyConn_Spatial4_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4OnlyConnEnvChars)
km_EnvChars_OnlyConn_Spatial4_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4OnlyConnEnvChars)

km_EnvChars_OnlyAcc_Spatial4_3 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4OnlyAccEnvChars)
km_EnvChars_OnlyAcc_Spatial4_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4OnlyAccEnvChars)
km_EnvChars_OnlyAcc_Spatial4_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4OnlyAccEnvChars)
km_EnvChars_OnlyAcc_Spatial4_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4OnlyAccEnvChars)
km_EnvChars_OnlyAcc_Spatial4_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4OnlyAccEnvChars)


# %%
km_EnvChars_OnlyAcc_Spatial5_3 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess5OnlyAccEnvChars)
km_EnvChars_OnlyAcc_Spatial5_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess5OnlyAccEnvChars)
km_EnvChars_OnlyAcc_Spatial5_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess5OnlyAccEnvChars)
km_EnvChars_OnlyAcc_Spatial5_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess5OnlyAccEnvChars)


# %%
spatialEnvAllSocialTess = spatialEnvSocialTess.copy()
spatialEnvAllSocialTess['clusters_form_set1_social_conn_EP_all_5'] = km_EnvAll_OnlyConn_Spatial1_5.labels_
spatialEnvAllSocialTess['clusters_form_set1_social_conn_EP_all_6'] = km_EnvAll_OnlyConn_Spatial1_6.labels_
spatialEnvAllSocialTess['clusters_form_set1_social_conn_EP_all_9'] = km_EnvAll_OnlyConn_Spatial1_9.labels_
spatialEnvAllSocialTess['clusters_form_set1_social_conn_EP_all_10'] = km_EnvAll_OnlyConn_Spatial1_10.labels_

spatialEnvAllSocialTess['clusters_form_set1_social_accessScore_EP_all_5'] = km_EnvAll_OnlyAcc_Spatial1_5.labels_
spatialEnvAllSocialTess['clusters_form_set1_social_accessScore_EP_all_6'] = km_EnvAll_OnlyAcc_Spatial1_6.labels_

spatialEnvAllSocialTess['clusters_form_set2_social_conn_EP_all_4'] = km_EnvAll_OnlyConn_Spatial2_4.labels_

spatialEnvAllSocialTess['clusters_form_set2_social_accessScore_EP_all_4'] = km_EnvAll_OnlyAcc_Spatial2_4.labels_
spatialEnvAllSocialTess['clusters_form_set2_social_accessScore_EP_all_5'] = km_EnvAll_OnlyAcc_Spatial2_5.labels_
spatialEnvAllSocialTess['clusters_form_set2_social_accessScore_EP_all_6'] = km_EnvAll_OnlyAcc_Spatial2_6.labels_
spatialEnvAllSocialTess['clusters_form_set2_social_accessScore_EP_all_8'] = km_EnvAll_OnlyAcc_Spatial2_8.labels_
spatialEnvAllSocialTess['clusters_form_set2_social_accessScore_EP_all_13'] = km_EnvAll_OnlyAcc_Spatial2_13.labels_

spatialEnvAllSocialTess['clusters_form_set3_social_conn_EP_all_4'] = km_EnvAll_OnlyConn_Spatial3_4.labels_
spatialEnvAllSocialTess['clusters_form_set3_social_conn_EP_all_6'] = km_EnvAll_OnlyConn_Spatial3_6.labels_

spatialEnvAllSocialTess['clusters_form_set3_social_accessScore_EP_all_4'] = km_EnvAll_OnlyAcc_Spatial3_4.labels_
spatialEnvAllSocialTess['clusters_form_set3_social_accessScore_EP_all_5'] = km_EnvAll_OnlyAcc_Spatial3_5.labels_
spatialEnvAllSocialTess['clusters_form_set3_social_accessScore_EP_all_6'] = km_EnvAll_OnlyAcc_Spatial3_6.labels_
spatialEnvAllSocialTess['clusters_form_set3_social_accessScore_EP_all_9'] = km_EnvAll_OnlyAcc_Spatial3_9.labels_

spatialEnvAllSocialTess['clusters_form_set4_social_conn_EP_all_5'] = km_EnvAll_OnlyConn_Spatial4_5.labels_
spatialEnvAllSocialTess['clusters_form_set4_social_conn_EP_all_6'] = km_EnvAll_OnlyConn_Spatial4_6.labels_
spatialEnvAllSocialTess['clusters_form_set4_social_conn_EP_all_7'] = km_EnvAll_OnlyConn_Spatial4_7.labels_

spatialEnvAllSocialTess['clusters_form_set4_social_accessScore_EP_all_5'] = km_EnvAll_OnlyAcc_Spatial4_5.labels_
spatialEnvAllSocialTess['clusters_form_set4_social_accessScore_EP_all_7'] = km_EnvAll_OnlyAcc_Spatial4_7.labels_
spatialEnvAllSocialTess['clusters_form_set4_social_accessScore_EP_all_10'] = km_EnvAll_OnlyAcc_Spatial4_10.labels_

spatialEnvAllSocialTess['clusters_form_set5_social_conn_EP_all_4'] = km_EnvAll_OnlyConn_Spatial5_4.labels_
spatialEnvAllSocialTess['clusters_form_set5_social_conn_EP_all_7'] = km_EnvAll_OnlyConn_Spatial5_7.labels_
spatialEnvAllSocialTess['clusters_form_set5_social_conn_EP_all_11'] = km_EnvAll_OnlyConn_Spatial5_11.labels_

spatialEnvAllSocialTess['clusters_form_set5_social_accessScore_EP_all_4'] = km_EnvAll_OnlyAcc_Spatial5_4.labels_
spatialEnvAllSocialTess['clusters_form_set5_social_accessScore_EP_all_11'] = km_EnvAll_OnlyAcc_Spatial5_11.labels_
spatialEnvAllSocialTess['clusters_form_set5_social_accessScore_EP_all_15'] = km_EnvAll_OnlyAcc_Spatial5_15.labels_

spatialEnvAllSocialTess.to_parquet(folder + "finalSpatialClusteringEnvPredAll.pq")

# %%
spatialEnvScoresSocialTess = spatialEnvSocialTess.copy()
spatialEnvScoresSocialTess['clusters_form_all_social_conn_EP_scores_3'] = km_EnvScores_OnlyConn_SpatialAll_3.labels_
spatialEnvScoresSocialTess['clusters_form_all_social_conn_EP_scores_6'] = km_EnvScores_OnlyConn_SpatialAll_6.labels_
spatialEnvScoresSocialTess['clusters_form_all_social_conn_EP_scores_9'] = km_EnvScores_OnlyConn_SpatialAll_9.labels_
spatialEnvScoresSocialTess['clusters_form_all_social_conn_EP_scores_12'] = km_EnvScores_OnlyConn_SpatialAll_12.labels_

spatialEnvScoresSocialTess['clusters_form_all_social_accessScore_EP_scores_3'] = km_EnvScores_OnlyAcc_SpatialAll_3.labels_
spatialEnvScoresSocialTess['clusters_form_all_social_accessScore_EP_scores_8'] = km_EnvScores_OnlyAcc_SpatialAll_8.labels_
spatialEnvScoresSocialTess['clusters_form_all_social_accessScore_EP_scores_11'] = km_EnvScores_OnlyAcc_SpatialAll_11.labels_

spatialEnvScoresSocialTess['clusters_form_set1_social_conn_EP_scores_5'] = km_EnvScores_OnlyConn_Spatial1_5.labels_
spatialEnvScoresSocialTess['clusters_form_set1_social_conn_EP_scores_8'] = km_EnvScores_OnlyConn_Spatial1_8.labels_
spatialEnvScoresSocialTess['clusters_form_set1_social_conn_EP_scores_11'] = km_EnvScores_OnlyConn_Spatial1_11.labels_
spatialEnvScoresSocialTess['clusters_form_set1_social_conn_EP_scores_13'] = km_EnvScores_OnlyConn_Spatial1_13.labels_

spatialEnvScoresSocialTess['clusters_form_set1_social_accessScore_EP_scores_5'] = km_EnvScores_OnlyAcc_Spatial1_5.labels_
spatialEnvScoresSocialTess['clusters_form_set1_social_accessScore_EP_scores_7'] = km_EnvScores_OnlyAcc_Spatial1_7.labels_
spatialEnvScoresSocialTess['clusters_form_set1_social_accessScore_EP_scores_10'] = km_EnvScores_OnlyAcc_Spatial1_10.labels_
spatialEnvScoresSocialTess['clusters_form_set1_social_accessScore_EP_scores_12'] = km_EnvScores_OnlyAcc_Spatial1_12.labels_

spatialEnvScoresSocialTess['clusters_form_set2_social_conn_EP_scores_3'] = km_EnvScores_OnlyConn_Spatial2_3.labels_
spatialEnvScoresSocialTess['clusters_form_set2_social_conn_EP_scores_6'] = km_EnvScores_OnlyConn_Spatial2_6.labels_
spatialEnvScoresSocialTess['clusters_form_set2_social_conn_EP_scores_10'] = km_EnvScores_OnlyConn_Spatial2_10.labels_
spatialEnvScoresSocialTess['clusters_form_set2_social_conn_EP_scores_12'] = km_EnvScores_OnlyConn_Spatial2_12.labels_

spatialEnvScoresSocialTess['clusters_form_set2_social_accessScore_EP_scores_3'] = km_EnvScores_OnlyAcc_Spatial2_3.labels_
spatialEnvScoresSocialTess['clusters_form_set2_social_accessScore_EP_scores_6'] = km_EnvScores_OnlyAcc_Spatial2_6.labels_
spatialEnvScoresSocialTess['clusters_form_set2_social_accessScore_EP_scores_8'] = km_EnvScores_OnlyAcc_Spatial2_8.labels_
spatialEnvScoresSocialTess['clusters_form_set2_social_accessScore_EP_scores_11'] = km_EnvScores_OnlyAcc_Spatial2_11.labels_

spatialEnvScoresSocialTess['clusters_form_set3_social_conn_EP_scores_3'] = km_EnvScores_OnlyConn_Spatial3_3.labels_
spatialEnvScoresSocialTess['clusters_form_set3_social_conn_EP_scores_6'] = km_EnvScores_OnlyConn_Spatial3_6.labels_
spatialEnvScoresSocialTess['clusters_form_set3_social_conn_EP_scores_7'] = km_EnvScores_OnlyConn_Spatial3_7.labels_
spatialEnvScoresSocialTess['clusters_form_set3_social_conn_EP_scores_12'] = km_EnvScores_OnlyConn_Spatial3_12.labels_

spatialEnvScoresSocialTess['clusters_form_set3_social_accessScore_EP_scores_4'] = km_EnvScores_OnlyAcc_Spatial3_4.labels_
spatialEnvScoresSocialTess['clusters_form_set3_social_accessScore_EP_scores_5'] = km_EnvScores_OnlyAcc_Spatial3_5.labels_
spatialEnvScoresSocialTess['clusters_form_set3_social_accessScore_EP_scores_6'] = km_EnvScores_OnlyAcc_Spatial3_6.labels_
spatialEnvScoresSocialTess['clusters_form_set3_social_accessScore_EP_scores_7'] = km_EnvScores_OnlyAcc_Spatial3_7.labels_

spatialEnvScoresSocialTess.to_parquet(folder + "finalSpatialClusteringEnvPredScores.pq")

# %%
spatialEnvCharsSocialTess = spatialEnvSocialTess.copy()
spatialEnvCharsSocialTess['clusters_form_set1_social_conn_EP_chars_3'] = km_EnvChars_OnlyConn_Spatial1_3.labels_
spatialEnvCharsSocialTess['clusters_form_set1_social_conn_EP_chars_4'] = km_EnvChars_OnlyConn_Spatial1_4.labels_
spatialEnvCharsSocialTess['clusters_form_set1_social_conn_EP_chars_5'] = km_EnvChars_OnlyConn_Spatial1_5.labels_
spatialEnvCharsSocialTess['clusters_form_set1_social_conn_EP_chars_6'] = km_EnvChars_OnlyConn_Spatial1_6.labels_

spatialEnvCharsSocialTess['clusters_form_set1_social_accessScore_EP_chars_3'] = km_EnvChars_OnlyAcc_Spatial1_3.labels_
spatialEnvCharsSocialTess['clusters_form_set1_social_accessScore_EP_chars_4'] = km_EnvChars_OnlyAcc_Spatial1_4.labels_
spatialEnvCharsSocialTess['clusters_form_set1_social_accessScore_EP_chars_5'] = km_EnvChars_OnlyAcc_Spatial1_5.labels_
spatialEnvCharsSocialTess['clusters_form_set1_social_accessScore_EP_chars_7'] = km_EnvChars_OnlyAcc_Spatial1_7.labels_

spatialEnvCharsSocialTess['clusters_form_set3_social_conn_EP_chars_3'] = km_EnvChars_OnlyConn_Spatial3_3.labels_
spatialEnvCharsSocialTess['clusters_form_set3_social_conn_EP_chars_4'] = km_EnvChars_OnlyConn_Spatial3_4.labels_
spatialEnvCharsSocialTess['clusters_form_set3_social_conn_EP_chars_5'] = km_EnvChars_OnlyConn_Spatial3_5.labels_
spatialEnvCharsSocialTess['clusters_form_set3_social_conn_EP_chars_12'] = km_EnvChars_OnlyConn_Spatial3_12.labels_

spatialEnvCharsSocialTess['clusters_form_set3_social_accessScore_EP_chars_3'] = km_EnvChars_OnlyAcc_Spatial3_3.labels_
spatialEnvCharsSocialTess['clusters_form_set3_social_accessScore_EP_chars_4'] = km_EnvChars_OnlyAcc_Spatial3_4.labels_
spatialEnvCharsSocialTess['clusters_form_set3_social_accessScore_EP_chars_5'] = km_EnvChars_OnlyAcc_Spatial3_5.labels_
spatialEnvCharsSocialTess['clusters_form_set3_social_accessScore_EP_chars_6'] = km_EnvChars_OnlyAcc_Spatial3_6.labels_
spatialEnvCharsSocialTess['clusters_form_set3_social_accessScore_EP_chars_7'] = km_EnvChars_OnlyAcc_Spatial3_7.labels_
spatialEnvCharsSocialTess['clusters_form_set3_social_accessScore_EP_chars_12'] = km_EnvChars_OnlyAcc_Spatial3_12.labels_

spatialEnvCharsSocialTess['clusters_form_set4_social_conn_EP_chars_3'] = km_EnvChars_OnlyConn_Spatial4_3.labels_
spatialEnvCharsSocialTess['clusters_form_set4_social_conn_EP_chars_4'] = km_EnvChars_OnlyConn_Spatial4_4.labels_
spatialEnvCharsSocialTess['clusters_form_set4_social_conn_EP_chars_5'] = km_EnvChars_OnlyConn_Spatial4_5.labels_
spatialEnvCharsSocialTess['clusters_form_set4_social_conn_EP_chars_6'] = km_EnvChars_OnlyConn_Spatial4_6.labels_
spatialEnvCharsSocialTess['clusters_form_set4_social_conn_EP_chars_8'] = km_EnvChars_OnlyConn_Spatial4_8.labels_
spatialEnvCharsSocialTess['clusters_form_set4_social_conn_EP_chars_10'] = km_EnvChars_OnlyConn_Spatial4_10.labels_
spatialEnvCharsSocialTess['clusters_form_set4_social_conn_EP_chars_12'] = km_EnvChars_OnlyConn_Spatial4_12.labels_

spatialEnvCharsSocialTess['clusters_form_set4_social_accessScore_EP_chars_3'] = km_EnvChars_OnlyAcc_Spatial4_3.labels_
spatialEnvCharsSocialTess['clusters_form_set4_social_accessScore_EP_chars_4'] = km_EnvChars_OnlyAcc_Spatial4_4.labels_
spatialEnvCharsSocialTess['clusters_form_set4_social_accessScore_EP_chars_5'] = km_EnvChars_OnlyAcc_Spatial4_5.labels_
spatialEnvCharsSocialTess['clusters_form_set4_social_accessScore_EP_chars_7'] = km_EnvChars_OnlyAcc_Spatial4_7.labels_
spatialEnvCharsSocialTess['clusters_form_set4_social_accessScore_EP_chars_10'] = km_EnvChars_OnlyAcc_Spatial4_10.labels_

spatialEnvCharsSocialTess['clusters_form_set5_social_accessScore_EP_chars_3'] = km_EnvChars_OnlyAcc_Spatial5_3.labels_
spatialEnvCharsSocialTess['clusters_form_set5_social_accessScore_EP_chars_4'] = km_EnvChars_OnlyAcc_Spatial5_4.labels_
spatialEnvCharsSocialTess['clusters_form_set5_social_accessScore_EP_chars_10'] = km_EnvChars_OnlyAcc_Spatial5_10.labels_
spatialEnvCharsSocialTess['clusters_form_set5_social_accessScore_EP_chars_12'] = km_EnvChars_OnlyAcc_Spatial5_12.labels_

spatialEnvCharsSocialTess.to_parquet(folder + "finalSpatialClusteringEnvPredChars.pq")

# %% [markdown]
# #### Plots (final)
# Come back to this later

# %% [markdown]
# #### Plots (semi-final)

# %% [markdown]
# ##### Env Pred All

# %% [markdown]
# ###### Set 1

# %%
# Set 1 - connectivity
ax = spatialEnvAllSocialTess.plot('clusters_form_set1_social_conn_EP_all_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (all) Clusters ETCs (5)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set1_social_conn_EP_all_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (all) Clusters ETCs (6)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set1_social_conn_EP_all_9', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (all) Clusters ETCs (9)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set1_social_conn_EP_all_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (all) Clusters ETCs (10)")


# %%

# Set 1 - access score
ax = spatialEnvAllSocialTess.plot('clusters_form_set1_social_accessScore_EP_all_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only access scores) + Environmental Prediction (all) Clusters ETCs (5)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set1_social_accessScore_EP_all_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only access scores) + Environmental Prediction (all) Clusters ETCs (6)")



# %% [markdown]
# ###### Set 2

# %%
# Set 2 - connectivity
ax = spatialEnvAllSocialTess.plot('clusters_form_set2_social_conn_EP_all_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 2) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (all) Clusters ETCs (4)")


# %%

# Set 2 - access score
ax = spatialEnvAllSocialTess.plot('clusters_form_set2_social_accessScore_EP_all_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 2) + Social Sustainability Indicators (only access scores) + Environmental Prediction (all) Clusters ETCs (4)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set2_social_accessScore_EP_all_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 2) + Social Sustainability Indicators (only access scores) + Environmental Prediction (all) Clusters ETCs (5)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set2_social_accessScore_EP_all_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 2) + Social Sustainability Indicators (only access scores) + Environmental Prediction (all) Clusters ETCs (6)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set2_social_accessScore_EP_all_8', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 2) + Social Sustainability Indicators (only access scores) + Environmental Prediction (all) Clusters ETCs (8)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set2_social_accessScore_EP_all_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 2) + Social Sustainability Indicators (only access scores) + Environmental Prediction (all) Clusters ETCs (13)")



# %% [markdown]
# ###### Set 3

# %%
# Set 3 - connectivity
ax = spatialEnvAllSocialTess.plot('clusters_form_set3_social_conn_EP_all_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (all) Clusters ETCs (4)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set3_social_conn_EP_all_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (all) Clusters ETCs (6)")


# %%

# Set 3 - access score
ax = spatialEnvAllSocialTess.plot('clusters_form_set3_social_accessScore_EP_all_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only access scores) + Environmental Prediction (all) Clusters ETCs (4)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set3_social_accessScore_EP_all_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only access scores) + Environmental Prediction (all) Clusters ETCs (5)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set3_social_accessScore_EP_all_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only access scores) + Environmental Prediction (all) Clusters ETCs (6)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set3_social_accessScore_EP_all_9', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only access scores) + Environmental Prediction (all) Clusters ETCs (9)")



# %% [markdown]
# ###### Set 4

# %%
# Set 4 - connectivity
ax = spatialEnvAllSocialTess.plot('clusters_form_set4_social_conn_EP_all_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 4) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (all) Clusters ETCs (5)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set4_social_conn_EP_all_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 4) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (all) Clusters ETCs (6)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set4_social_conn_EP_all_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 4) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (all) Clusters ETCs (7)")


# %%
# Set 4 - access score
ax = spatialEnvAllSocialTess.plot('clusters_form_set4_social_accessScore_EP_all_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 4) + Social Sustainability Indicators (only access scores) + Environmental Prediction (all) Clusters ETCs (5)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set4_social_accessScore_EP_all_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 4) + Social Sustainability Indicators (only access scores) + Environmental Prediction (all) Clusters ETCs (7)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set4_social_accessScore_EP_all_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 4) + Social Sustainability Indicators (only access scores) + Environmental Prediction (all) Clusters ETCs (10)")


# %% [markdown]
# ###### Set 5

# %%
# Set 5 - connectivity
ax = spatialEnvAllSocialTess.plot('clusters_form_set5_social_conn_EP_all_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 5) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (all) Clusters ETCs (4)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set5_social_conn_EP_all_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 5) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (all) Clusters ETCs (7)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set5_social_conn_EP_all_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 5) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (all) Clusters ETCs (11)")


# %%

# Set 5 - access score
ax = spatialEnvAllSocialTess.plot('clusters_form_set5_social_accessScore_EP_all_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 5) + Social Sustainability Indicators (only access scores) + Environmental Prediction (all) Clusters ETCs (4)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set5_social_accessScore_EP_all_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 5) + Social Sustainability Indicators (only access scores) + Environmental Prediction (all) Clusters ETCs (11)")

ax = spatialEnvAllSocialTess.plot('clusters_form_set5_social_accessScore_EP_all_15', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 5) + Social Sustainability Indicators (only access scores) + Environmental Prediction (all) Clusters ETCs (15)")


# %% [markdown]
# ##### Env Pred Scores

# %% [markdown]
# ###### All Spatial

# %%
# All Spatial - connectivity
ax = spatialEnvScoresSocialTess.plot('clusters_form_all_social_conn_EP_scores_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (All selected) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (scores) Clusters ETCs (5)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_all_social_conn_EP_scores_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (All selected) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (scores) Clusters ETCs (6)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_all_social_conn_EP_scores_9', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (All selected) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (scores) Clusters ETCs (9)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_all_social_conn_EP_scores_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (All selected) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (scores) Clusters ETCs (12)")


# %%

# All Spatial - access score
ax = spatialEnvScoresSocialTess.plot('clusters_form_all_social_accessScore_EP_scores_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (All selected) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Scores) Clusters ETCs (3)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_all_social_accessScore_EP_scores_8', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (All selected) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Scores) Clusters ETCs (8)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_all_social_accessScore_EP_scores_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (All selected) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Scores) Clusters ETCs (11)")



# %% [markdown]
# ###### Set 1

# %%

# Set 1 - connectivity
ax = spatialEnvScoresSocialTess.plot('clusters_form_set1_social_conn_EP_scores_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Scores) Clusters ETCs (5)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_set1_social_conn_EP_scores_8', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Scores) Clusters ETCs (8)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_set1_social_conn_EP_scores_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Scores) Clusters ETCs (11)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_set1_social_conn_EP_scores_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Scores) Clusters ETCs (13)")


# %%

# Set 1 - access score
ax = spatialEnvScoresSocialTess.plot('clusters_form_set1_social_accessScore_EP_scores_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Scores) Clusters ETCs (5)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_set1_social_accessScore_EP_scores_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Scores) Clusters ETCs (7)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_set1_social_accessScore_EP_scores_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Scores) Clusters ETCs (10)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_set1_social_accessScore_EP_scores_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Scores) Clusters ETCs (12)")



# %% [markdown]
# ###### Set 2

# %%

# Set 2 - connectivity
ax = spatialEnvScoresSocialTess.plot('clusters_form_set2_social_conn_EP_scores_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 2) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Scores) Clusters ETCs (3)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_set2_social_conn_EP_scores_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 2) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Scores) Clusters ETCs (6)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_set2_social_conn_EP_scores_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 2) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Scores) Clusters ETCs (10)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_set2_social_conn_EP_scores_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 2) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Scores) Clusters ETCs (12)")


# %%

# Set 2 - access score
ax = spatialEnvScoresSocialTess.plot('clusters_form_set2_social_accessScore_EP_scores_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 2) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Scores) Clusters ETCs (3)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_set2_social_accessScore_EP_scores_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 2) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Scores) Clusters ETCs (6)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_set2_social_accessScore_EP_scores_8', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 2) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Scores) Clusters ETCs (8)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_set2_social_accessScore_EP_scores_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 2) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Scores) Clusters ETCs (11)")



# %% [markdown]
# ###### Set 3

# %%

# Set 3 - connectivity
ax = spatialEnvScoresSocialTess.plot('clusters_form_set3_social_conn_EP_scores_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Scores) Clusters ETCs (3)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_set3_social_conn_EP_scores_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Scores) Clusters ETCs (6)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_set3_social_conn_EP_scores_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Scores) Clusters ETCs (7)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_set3_social_conn_EP_scores_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Scores) Clusters ETCs (12)")


# %%

# Set 3 - access score
ax = spatialEnvScoresSocialTess.plot('clusters_form_set3_social_accessScore_EP_scores_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Scores) Clusters ETCs (4)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_set3_social_accessScore_EP_scores_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Scores) Clusters ETCs (5)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_set3_social_accessScore_EP_scores_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Scores) Clusters ETCs (6)")

ax = spatialEnvScoresSocialTess.plot('clusters_form_set3_social_accessScore_EP_scores_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Scores) Clusters ETCs (7)")


# %% [markdown]
# ##### Env Pred Character

# %% [markdown]
# ###### Set 1

# %%
# Set 1 - connectivity
ax = spatialEnvCharsSocialTess.plot('clusters_form_set1_social_conn_EP_chars_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Character) Clusters ETCs (3)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set1_social_conn_EP_chars_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Character) Clusters ETCs (4)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set1_social_conn_EP_chars_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Character) Clusters ETCs (5)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set1_social_conn_EP_chars_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Character) Clusters ETCs (6)")


# %%

# Set 1 - access score
ax = spatialEnvCharsSocialTess.plot('clusters_form_set1_social_accessScore_EP_chars_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (3)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set1_social_accessScore_EP_chars_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (4)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set1_social_accessScore_EP_chars_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (5)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set1_social_accessScore_EP_chars_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 1) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (7)")


# %% [markdown]
# ###### Set 3

# %%

# Set 3 - connectivity
ax = spatialEnvCharsSocialTess.plot('clusters_form_set3_social_conn_EP_chars_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Character) Clusters ETCs (3)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set3_social_conn_EP_chars_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Character) Clusters ETCs (4)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set3_social_conn_EP_chars_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Character) Clusters ETCs (5)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set3_social_conn_EP_chars_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Character) Clusters ETCs (12)")


# %%

# Set 3 - access score
ax = spatialEnvCharsSocialTess.plot('clusters_form_set3_social_accessScore_EP_chars_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (3)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set3_social_accessScore_EP_chars_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (4)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set3_social_accessScore_EP_chars_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (5)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set3_social_accessScore_EP_chars_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (6)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set3_social_accessScore_EP_chars_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (7)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set3_social_accessScore_EP_chars_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 3) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (12)")


# %% [markdown]
# ###### Set 4

# %%


# Set 4 - connectivity
ax = spatialEnvCharsSocialTess.plot('clusters_form_set4_social_conn_EP_chars_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 4) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Character) Clusters ETCs (3)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set4_social_conn_EP_chars_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 4) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Character) Clusters ETCs (4)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set4_social_conn_EP_chars_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 4) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Character) Clusters ETCs (5)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set4_social_conn_EP_chars_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 4) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Character) Clusters ETCs (6)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set4_social_conn_EP_chars_8', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 4) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Character) Clusters ETCs (8)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set4_social_conn_EP_chars_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 4) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Character) Clusters ETCs (10)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set4_social_conn_EP_chars_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 4) + Social Sustainability Indicators (only centrality values) + Environmental Prediction (Character) Clusters ETCs (12)")


# %%

# Set 4 - access score
ax = spatialEnvCharsSocialTess.plot('clusters_form_set4_social_accessScore_EP_chars_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 4) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (3)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set4_social_accessScore_EP_chars_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 4) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (4)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set4_social_accessScore_EP_chars_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 4) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (5)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set4_social_accessScore_EP_chars_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 4) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (7)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set4_social_accessScore_EP_chars_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 4) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (10)")



# %% [markdown]
# ###### Set 5

# %%
# Set 5 - access score
ax = spatialEnvCharsSocialTess.plot('clusters_form_set5_social_accessScore_EP_chars_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 5) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (3)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set5_social_accessScore_EP_chars_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 5) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (4)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set5_social_accessScore_EP_chars_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 5) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (10)")

ax = spatialEnvCharsSocialTess.plot('clusters_form_set5_social_accessScore_EP_chars_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (Set 5) + Social Sustainability Indicators (only access scores) + Environmental Prediction (Character) Clusters ETCs (12)")


# %% [markdown]
# #### Cluster Analysis Old

# %% [markdown]
# ##### Set 1

# %%
cg_Full1 = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Full1.fit(dataTess1)
ax1 = cg_Full1.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax1.set_title("Spatial form (full) + Social Sustainability Indicators (all) + Environmental Prediction (all) Silhouette score")

# %% [markdown]
# Clearly all spatial patterns add too much dimensionality for the clustering to be easily differentiable, thus information differences between clusters might not be clear enough to infer anything.

# %% [markdown]
# ##### Set 2

# %%
cg_Full2 = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Full2.fit(dataTess2)
ax2 = cg_Full2.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (selected) + Social Sustainability Indicators (all) + Environmental Prediction (all) Silhouette score")

# %%
ax2 = cg_Full2.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax2.set_title("Spatial form (selected) + Social Sustainability Indicators (all) + Environmental Prediction (all) Silhouette score")
ax2.axvline(6, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(10, color=c[4], ymin=.02, ymax=.98)
ax2.axvline(13, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# For Spatial form (selected) + Social Sustainability Indicators (all) + Environmental Prediction (all), there are peaks of interest at 6 clusters, 10 clusters, and 13 clusters according to the silhouette score.

# %% [markdown]
# ##### Set 3

# %%
cg_Full3 = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Full3.fit(dataTess3)
ax3 = cg_Full3.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax3.set_title("Spatial form (selected) + Social Sustainability Indicators (no population) + Environmental Prediction (all) Silhouette score")

# %%
ax3 = cg_Full3.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax3.set_title("Spatial form (selected) + Social Sustainability Indicators (no population) + Environmental Prediction (all) Silhouette score")
ax3.axvline(4, color=c[4], ymin=.02, ymax=.98)
ax3.axvline(6, color=c[4], ymin=.02, ymax=.98)
ax3.axvline(10, color=c[4], ymin=.02, ymax=.98)
ax3.axvline(13, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# Losing population data, which just added more info to the housing data, seems to have reduced the ability of the clusters to be differentiable. However still clusters at 4, 6, 10, and 13 are of interest.

# %% [markdown]
# ##### Set 4

# %%
cg_Full4 = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Full4.fit(dataTess4)
ax4 = cg_Full4.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax4.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop and no amenity diversity) + Environmental Prediction (all) Silhouette score")

# %%
ax4 = cg_Full4.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax4.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop and no amenity diversity) + Environmental Prediction (all) Silhouette score")
ax4.axvline(5, color=c[4], ymin=.02, ymax=.98)
ax4.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax4.axvline(10, color=c[4], ymin=.02, ymax=.98)
ax4.axvline(13, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# Removal of amenity diversity seemed to improve clustering, maybe this added dimensionality had a disproportionate impact on difficulty with clustering. This information is interesting to determining access of populations to different building types, but apparently it gets in the way of making conclusions about connectivity, shape, and env. pred.
# 
# Clusters of interest to investigate are 5, 7, 10, and 13 clusters. 

# %% [markdown]
# ##### Set 5

# %%
cg_Full5 = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Full5.fit(dataTess5)
ax5 = cg_Full5.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax5.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop, no amenity diversity, and no housing score) + Environmental Prediction (all) Silhouette score")

# %%
ax5 = cg_Full5.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax5.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop, no amenity diversity, and no housing score) + Environmental Prediction (all) Silhouette score")
ax5.axvline(5, color=c[4], ymin=.02, ymax=.98)
ax5.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax5.axvline(11, color=c[4], ymin=.02, ymax=.98)
ax5.axvline(14, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# Clusters at 5, 7, 11, and 14
# 

# %% [markdown]
# ##### Set 6

# %%
cg_Full6 = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Full6.fit(dataTess6)
ax6 = cg_Full6.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax6.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop, no amenity diversity, and no housing score) + Environmental Prediction (no heat or water score) Silhouette score")

# %%
ax6 = cg_Full6.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax6.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop, no amenity diversity, and no housing score) + Environmental Prediction (no heat or water score) Silhouette score")
ax6.axvline(4, color=c[4], ymin=.02, ymax=.98)
# ax6.axvline(7, color=c[4], ymin=.02, ymax=.98)
ax6.axvline(10, color=c[4], ymin=.02, ymax=.98)
ax6.axvline(13, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# Peaks of clustering happen at 4, 10, and 13 clusters. However these are some of the least differntiable apart from the first clustergram.
# Will only look at the PCAs for set 2, 4, 5, and 6
# 

# %% [markdown]
# ##### Set 7

# %%
cg_Full7 = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Full7.fit(dataTess7)
ax7 = cg_Full7.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax7.set_title("Spatial form (selected) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Silhouette score")

# %%
ax7 = cg_Full7.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax7.set_title("Spatial form (selected) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Silhouette score")
ax7.axvline(12, color=c[4], ymin=.02, ymax=.98)
ax7.axvline(13, color=c[4], ymin=.02, ymax=.98)
ax7.axvline(18, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# The reduction in dimensionality that came from removing all the other aspects of environmental prediction is quite promising in terms of the number of clusters that could now be formed with relative discernability. However, it must be noted that the silhouette score is still quite low. Will try reducing dimensionality even further. 
# 
# Clusters of interest: 12, 13, and 18

# %% [markdown]
# ##### Set 8

# %%
cg_Full8 = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Full8.fit(dataTess8)
ax8 = cg_Full8.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax8.set_title("Spatial form (selected fewer) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Silhouette score")

# %%
ax8 = cg_Full8.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax8.set_title("Spatial form (selected fewer) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Silhouette score")
ax8.axvline(10, color=c[4], ymin=.02, ymax=.98)
ax8.axvline(12, color=c[4], ymin=.02, ymax=.98)
ax8.axvline(17, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# Clusters of interest at 10, 12, and 17 clusters. Still low silhouette score, trying one more combination.

# %% [markdown]
# ##### Set 9

# %%
cg_Full9 = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Full9.fit(dataTess9)
ax9 = cg_Full9.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax9.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Silhouette score")

# %%
ax9 = cg_Full9.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax9.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Silhouette score")
ax9.axvline(3, color=c[4], ymin=.02, ymax=.98)
ax9.axvline(12, color=c[4], ymin=.02, ymax=.98)
ax9.axvline(17, color=c[4], ymin=.02, ymax=.98)
ax9.axvline(20, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# With the original selection clusters and just the score, it is clear that the clusters are fairly discernible but not ideal, especially since most discernible clusters occur at 3 clusters. The clusters will be done at 3, 12, 17, and 20

# %% [markdown]
# ##### Set 10

# %%
cg_Full10 = Clustergram(range(1, 50), backend='sklearn', n_init=30, random_state=12345)
cg_Full10.fit(dataTess10)
ax10 = cg_Full10.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax10.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (no scoring) Silhouette score")

# %%
ax10 = cg_Full10.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
ax10.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (no scoring) Silhouette score")
ax10.axvline(9, color=c[4], ymin=.02, ymax=.98)
ax10.axvline(12, color=c[4], ymin=.02, ymax=.98)
ax10.axvline(14, color=c[4], ymin=.02, ymax=.98)
ax10.axvline(18, color=c[4], ymin=.02, ymax=.98)

# %% [markdown]
# Set 8 seems to have the best clustering trends. However for set 10 the clusters of interest happen at 9, 12, 14, and 18 clusters.

# %% [markdown]
# ##### PCA Clustergram Plots

# %%
c = ugg.COLORS
ax = cg_Full2.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramFullSet2.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# clusters at 8, 11, and 12

# %%
cg_labels = cg_Full2.labels
cg_labels.columns = cg_labels.columns.astype(str)
cg_labels.to_parquet(folder + 'clustergram_FullSet2_labels.pq')


# %%
c = ugg.COLORS
ax = cg_Full3.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(9, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramFullSet3.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# 4 6 10 13 (silhouette score)
# 11 12 18 (clustergram)

# %%
cg_labels = cg_Full3.labels
cg_labels.columns = cg_labels.columns.astype(str)
cg_labels.to_parquet(folder + 'clustergram_FullSet3_labels.pq')

# %%
c = ugg.COLORS
ax = cg_Full4.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(9, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramFullSet4.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# clusters at 9, 11, 12, and 18

# %%
cg_labels = cg_Full4.labels
cg_labels.columns = cg_labels.columns.astype(str)
cg_labels.to_parquet(folder + 'clustergram_FullSet4_labels.pq')

# %%
c = ugg.COLORS
ax = cg_Full5.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
ax.axvline(10, color=c[4], ymin=.02, ymax=.98)
# plt.savefig(folder + "cgramFullSet5.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# Clusters at 10, 11, and 12

# %%
cg_labels = cg_Full5.labels
cg_labels.columns = cg_labels.columns.astype(str)
cg_labels.to_parquet(folder + 'clustergram_FullSet5_labels.pq')

# %%
c = ugg.COLORS
ax = cg_Full6.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
# plt.savefig(folder + "cgramFullSet6.png", dpi=300, bbox_inches="tight")

# %%
cg_labels = cg_Full6.labels
cg_labels.columns = cg_labels.columns.astype(str)
cg_labels.to_parquet(folder + 'clustergram_FullSet6_labels.pq')

# %% [markdown]
# 6, 7, 12 clusters.
# 
# This means for sets the following will be the clusters of interest:
# 
# - Set 2: 6, 8, 10, 11, 12, and 13
# - Set 3: 4, 6, 10, 11, 12, 13, and 18
# - Set 4: 5, 7, 9, 10, 11, 12, 13, and 18
# - Set 5: 5, 7, 10, 11, 12, and 14
# - Set 6: 4, 6, 7, 10, 12, and 13
# - Set 7: 5, 8, 11, 12, 13, and 18
# - Set 8: 6, 7, 9, 10, 12, 13, and 17
# - Set 9: 3, 4, 7, 9, 10, 12, 17 and 20 
# - Set 10: 6, 10, 12, 15, 14, and 18

# %%
c = ugg.COLORS
ax = cg_Full7.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramFullSet7.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# 5, 8, 11

# %%
cg_labels = cg_Full7.labels
cg_labels.columns = cg_labels.columns.astype(str)
cg_labels.to_parquet(folder + 'clustergram_FullSet7_labels.pq')

# %%
c = ugg.COLORS
ax = cg_Full8.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramFullSet8.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# 6, 7, 9, 13

# %%
cg_labels = cg_Full8.labels
cg_labels.columns = cg_labels.columns.astype(str)
cg_labels.to_parquet(folder + 'clustergram_FullSet8_labels.pq')

# %%
c = ugg.COLORS
ax = cg_Full9.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramFullSet9.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# 4, 7, 9, 10

# %%
cg_labels = cg_Full9.labels
cg_labels.columns = cg_labels.columns.astype(str)
cg_labels.to_parquet(folder + 'clustergram_FullSet9_labels.pq')

# %%
c = ugg.COLORS
ax = cg_Full10.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(7, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramFullSet10.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# 6, 10, 15

# %%
cg_labels = cg_Full10.labels
cg_labels.columns = cg_labels.columns.astype(str)
cg_labels.to_parquet(folder + 'clustergram_FullSet10_labels.pq')

# %% [markdown]
# #### Clustering Old

# %% [markdown]
# ##### KMeans Clustering

# %%
km2_6 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2)
km2_8 = MiniBatchKMeans(n_clusters=8, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2)
km2_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2)
km2_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2)
km2_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2)
km2_13 = MiniBatchKMeans(n_clusters=13, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2)

# %%
km3_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3)
km3_6 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3)
km3_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3)
km3_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3)
km3_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3)
km3_13 = MiniBatchKMeans(n_clusters=13, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3)
km3_18 = MiniBatchKMeans(n_clusters=18, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3)

# %%
km4_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4)
km4_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4)
km4_9 = MiniBatchKMeans(n_clusters=9, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4)
km4_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4)
km4_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4)
km4_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4)
km4_13 = MiniBatchKMeans(n_clusters=13, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4)
km4_18 = MiniBatchKMeans(n_clusters=18, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4)

# %%
%time km5_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess5)
%time km5_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess5)
%time km5_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess5)
%time km5_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess5)
%time km5_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess5)
%time km5_14 = MiniBatchKMeans(n_clusters=14, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess5)


# %%
%time km6_4 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess6)
%time km6_6 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess6)
%time km6_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess6)
%time km6_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess6)
%time km6_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess6)
%time km6_13 = MiniBatchKMeans(n_clusters=13, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess6)

# %%
%time km7_5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess7)
%time km7_8 = MiniBatchKMeans(n_clusters=8, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess7)
%time km7_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess7)
%time km7_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess7)
%time km7_13 = MiniBatchKMeans(n_clusters=13, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess7)
%time km7_18 = MiniBatchKMeans(n_clusters=18, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess7)

# %%
%time km8_6 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess8)
%time km8_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess8)
%time km8_9 = MiniBatchKMeans(n_clusters=9, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess8)
%time km8_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess8)
%time km8_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess8)
%time km8_13 = MiniBatchKMeans(n_clusters=13, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess8)
%time km8_17 = MiniBatchKMeans(n_clusters=17, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess8)

# %%
%time km9_3 = MiniBatchKMeans(n_clusters=3, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess9)
%time km9_4 = MiniBatchKMeans(n_clusters=4, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess9)
%time km9_7 = MiniBatchKMeans(n_clusters=7, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess9)
%time km9_9 = MiniBatchKMeans(n_clusters=9, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess9)
%time km9_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess9)
%time km9_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess9)
%time km9_17 = MiniBatchKMeans(n_clusters=17, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess9)
%time km9_20 = MiniBatchKMeans(n_clusters=20, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess9)

# %%
%time km10_6 = MiniBatchKMeans(n_clusters=6, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess10)
%time km10_9 = MiniBatchKMeans(n_clusters=9, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess10)
%time km10_10 = MiniBatchKMeans(n_clusters=10, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess10)
%time km10_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess10)
%time km10_14 = MiniBatchKMeans(n_clusters=14, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess10)
%time km10_15 = MiniBatchKMeans(n_clusters=15, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess10)
%time km10_18 = MiniBatchKMeans(n_clusters=18, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess10)

# %% [markdown]
# #### Old Plots

# %% [markdown]
# ##### Set 2 Plots

# %%
spatialEnvSocialTess2['clusters_form_social_envPred_6'] = km2_6.labels_
spatialEnvSocialTess2['clusters_form_social_envPred_8'] = km2_8.labels_
spatialEnvSocialTess2['clusters_form_social_envPred_10'] = km2_10.labels_
spatialEnvSocialTess2['clusters_form_social_envPred_11'] = km2_11.labels_
spatialEnvSocialTess2['clusters_form_social_envPred_12'] = km2_12.labels_
spatialEnvSocialTess2['clusters_form_social_envPred_13'] = km2_13.labels_

spatialEnvSocialTess2.to_parquet(folder + "spatialEnvSocialClustersTessSet2.pq")

# %%
ax = spatialEnvSocialTess2.plot('clusters_form_social_envPred_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (all) + Environmental Prediction (all) Clusters ETCs (6)")

ax = spatialEnvSocialTess2.plot('clusters_form_social_envPred_8', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (all) + Environmental Prediction (all) Clusters ETCs (8)")

ax = spatialEnvSocialTess2.plot('clusters_form_social_envPred_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (all) + Environmental Prediction (all) Clusters ETCs (10)")

ax = spatialEnvSocialTess2.plot('clusters_form_social_envPred_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (all) + Environmental Prediction (all) Clusters ETCs (11)")

ax = spatialEnvSocialTess2.plot('clusters_form_social_envPred_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (all) + Environmental Prediction (all) Clusters ETCs (12)")

ax = spatialEnvSocialTess2.plot('clusters_form_social_envPred_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (all) + Environmental Prediction (all) Clusters ETCs (13)")

# %% [markdown]
# For set 2, the clusters for 6 clusters, 10 clusters, and 13 clusters seem to be most aligned with the understanding of the separation of the area given the previous trends noted in clustering. However the population data might have caused issues with separation.

# %% [markdown]
# ##### Set 3 Plots

# %%
spatialEnvSocialTess3['clusters_form_social_envPred_4'] = km3_4.labels_
spatialEnvSocialTess3['clusters_form_social_envPred_6'] = km3_6.labels_
spatialEnvSocialTess3['clusters_form_social_envPred_10'] = km3_10.labels_
spatialEnvSocialTess3['clusters_form_social_envPred_11'] = km3_11.labels_
spatialEnvSocialTess3['clusters_form_social_envPred_12'] = km3_12.labels_
spatialEnvSocialTess3['clusters_form_social_envPred_13'] = km3_13.labels_
spatialEnvSocialTess3['clusters_form_social_envPred_18'] = km3_18.labels_

spatialEnvSocialTess3.to_parquet(folder + "spatialEnvSocialClustersTessSet3.pq")

# %%
ax = spatialEnvSocialTess3.plot('clusters_form_social_envPred_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no population) + Environmental Prediction (all) Clusters ETCs (4)")

ax = spatialEnvSocialTess3.plot('clusters_form_social_envPred_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no population) + Environmental Prediction (all) Clusters ETCs (6)")

ax = spatialEnvSocialTess3.plot('clusters_form_social_envPred_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no population) + Environmental Prediction (all) Clusters ETCs (10)")

ax = spatialEnvSocialTess3.plot('clusters_form_social_envPred_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no population) + Environmental Prediction (all) Clusters ETCs (11)")

ax = spatialEnvSocialTess3.plot('clusters_form_social_envPred_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no population) + Environmental Prediction (all) Clusters ETCs (12)")

ax = spatialEnvSocialTess3.plot('clusters_form_social_envPred_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no population) + Environmental Prediction (all) Clusters ETCs (13)")

ax = spatialEnvSocialTess3.plot('clusters_form_social_envPred_18', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no population) + Environmental Prediction (all) Clusters ETCs (18)")

# %% [markdown]
# 6, 11, and 13 clusters seemed to be most consistent with expectation. Might need to do a comparative analysis between the different scores. 

# %% [markdown]
# ##### Set 4 Plots

# %%
spatialEnvSocialTess4['clusters_form_social_envPred_5'] = km4_5.labels_
spatialEnvSocialTess4['clusters_form_social_envPred_7'] = km4_7.labels_
spatialEnvSocialTess4['clusters_form_social_envPred_9'] = km4_9.labels_
spatialEnvSocialTess4['clusters_form_social_envPred_10'] = km4_10.labels_
spatialEnvSocialTess4['clusters_form_social_envPred_11'] = km4_11.labels_
spatialEnvSocialTess4['clusters_form_social_envPred_12'] = km4_12.labels_
spatialEnvSocialTess4['clusters_form_social_envPred_13'] = km4_13.labels_
spatialEnvSocialTess4['clusters_form_social_envPred_18'] = km4_18.labels_

spatialEnvSocialTess4.to_parquet(folder + "spatialEnvSocialClustersTessSet4.pq")

# %%
ax = spatialEnvSocialTess4.plot('clusters_form_social_envPred_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop and no amenity diversity) + Environmental Prediction (all) Clusters ETCs (5)")

ax = spatialEnvSocialTess4.plot('clusters_form_social_envPred_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop and no amenity diversity) + Environmental Prediction (all) Clusters ETCs (7)")

ax = spatialEnvSocialTess4.plot('clusters_form_social_envPred_9', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop and no amenity diversity) + Environmental Prediction (all) Clusters ETCs (9)")

ax = spatialEnvSocialTess4.plot('clusters_form_social_envPred_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop and no amenity diversity) + Environmental Prediction (all) Clusters ETCs (10)")

ax = spatialEnvSocialTess4.plot('clusters_form_social_envPred_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop and no amenity diversity) + Environmental Prediction (all) Clusters ETCs (11)")

ax = spatialEnvSocialTess4.plot('clusters_form_social_envPred_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop and no amenity diversity) + Environmental Prediction (all) Clusters ETCs (12)")

ax = spatialEnvSocialTess4.plot('clusters_form_social_envPred_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop and no amenity diversity) + Environmental Prediction (all) Clusters ETCs (13)")

ax = spatialEnvSocialTess4.plot('clusters_form_social_envPred_18', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop and no amenity diversity) + Environmental Prediction (all) Clusters ETCs (18)")

# %% [markdown]
# 7, 10, and 12 clusters

# %% [markdown]
# ##### Set 5 plots

# %%
spatialEnvSocialTess5['clusters_form_social_envPred_5'] = km5_5.labels_
spatialEnvSocialTess5['clusters_form_social_envPred_7'] = km5_7.labels_
spatialEnvSocialTess5['clusters_form_social_envPred_10'] = km5_10.labels_
spatialEnvSocialTess5['clusters_form_social_envPred_11'] = km5_11.labels_
spatialEnvSocialTess5['clusters_form_social_envPred_12'] = km5_12.labels_
spatialEnvSocialTess5['clusters_form_social_envPred_14'] = km5_14.labels_

spatialEnvSocialTess5.to_parquet(folder + "spatialEnvSocialClustersTessSet5.pq")

# %%
ax = spatialEnvSocialTess5.plot('clusters_form_social_envPred_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop, no amenity diversity, and no housing score) + Environmental Prediction (all) Clusters ETCs (5)")

ax = spatialEnvSocialTess5.plot('clusters_form_social_envPred_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop, no amenity diversity, and no housing score) + Environmental Prediction (all) Clusters ETCs (7)")

ax = spatialEnvSocialTess5.plot('clusters_form_social_envPred_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop, no amenity diversity, and no housing score) + Environmental Prediction (all) Clusters ETCs (10)")

ax = spatialEnvSocialTess5.plot('clusters_form_social_envPred_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop, no amenity diversity, and no housing score) + Environmental Prediction (all) Clusters ETCs (11)")

ax = spatialEnvSocialTess5.plot('clusters_form_social_envPred_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop, no amenity diversity, and no housing score) + Environmental Prediction (all) Clusters ETCs (12)")

ax = spatialEnvSocialTess5.plot('clusters_form_social_envPred_14', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop, no amenity diversity, and no housing score) + Environmental Prediction (all) Clusters ETCs (14)")

# %% [markdown]
# 10  and 12 clusters seem to be most representative

# %% [markdown]
# ##### Set 6 Plots

# %%
spatialEnvSocialTess6['clusters_form_social_envPred_4'] = km6_4.labels_
spatialEnvSocialTess6['clusters_form_social_envPred_6'] = km6_6.labels_
spatialEnvSocialTess6['clusters_form_social_envPred_7'] = km6_7.labels_
spatialEnvSocialTess6['clusters_form_social_envPred_10'] = km6_10.labels_
spatialEnvSocialTess6['clusters_form_social_envPred_12'] = km6_12.labels_
spatialEnvSocialTess6['clusters_form_social_envPred_13'] = km6_13.labels_

spatialEnvSocialTess6.to_parquet(folder + "spatialEnvSocialClustersTessSet6.pq")

# %%
ax = spatialEnvSocialTess6.plot('clusters_form_social_envPred_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop, no amenity diversity, and no housing score) + Environmental Prediction (no heat or water score) Clusters ETCs (4)")

ax = spatialEnvSocialTess6.plot('clusters_form_social_envPred_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop, no amenity diversity, and no housing score) + Environmental Prediction (no heat or water score) Clusters ETCs (6)")

ax = spatialEnvSocialTess6.plot('clusters_form_social_envPred_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop, no amenity diversity, and no housing score) + Environmental Prediction (no heat or water score) Clusters ETCs (7)")

ax = spatialEnvSocialTess6.plot('clusters_form_social_envPred_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop, no amenity diversity, and no housing score) + Environmental Prediction (no heat or water score) Clusters ETCs (10)")

ax = spatialEnvSocialTess6.plot('clusters_form_social_envPred_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop, no amenity diversity, and no housing score) + Environmental Prediction (no heat or water score) Clusters ETCs (12)")

ax = spatialEnvSocialTess6.plot('clusters_form_social_envPred_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (no pop, no amenity diversity, and no housing score) + Environmental Prediction (no heat or water score) Clusters ETCs (13)")

# %% [markdown]
# ##### Set 7 Plots

# %%
spatialEnvSocialTess7['clusters_form_social_envPred_5'] = km7_5.labels_
spatialEnvSocialTess7['clusters_form_social_envPred_8'] = km7_8.labels_
spatialEnvSocialTess7['clusters_form_social_envPred_11'] = km7_11.labels_
spatialEnvSocialTess7['clusters_form_social_envPred_12'] = km7_12.labels_
spatialEnvSocialTess7['clusters_form_social_envPred_13'] = km7_13.labels_
spatialEnvSocialTess7['clusters_form_social_envPred_18'] = km7_18.labels_

spatialEnvSocialTess7.to_parquet(folder + "spatialEnvSocialClustersTessSet7.pq")

# %%
ax = spatialEnvSocialTess7.plot('clusters_form_social_envPred_5', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (5)")

ax = spatialEnvSocialTess7.plot('clusters_form_social_envPred_8', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (8)")

ax = spatialEnvSocialTess7.plot('clusters_form_social_envPred_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (11)")

ax = spatialEnvSocialTess7.plot('clusters_form_social_envPred_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (12)")

ax = spatialEnvSocialTess7.plot('clusters_form_social_envPred_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (13)")

ax = spatialEnvSocialTess7.plot('clusters_form_social_envPred_18', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (18)")

# %% [markdown]
# ##### Set 8 Plots

# %%
spatialEnvSocialTess8['clusters_form_social_envPred_6'] = km8_6.labels_
spatialEnvSocialTess8['clusters_form_social_envPred_7'] = km8_7.labels_
spatialEnvSocialTess8['clusters_form_social_envPred_9'] = km8_9.labels_
spatialEnvSocialTess8['clusters_form_social_envPred_10'] = km8_10.labels_
spatialEnvSocialTess8['clusters_form_social_envPred_12'] = km8_12.labels_
spatialEnvSocialTess8['clusters_form_social_envPred_13'] = km8_13.labels_
spatialEnvSocialTess8['clusters_form_social_envPred_17'] = km8_17.labels_

spatialEnvSocialTess8.to_parquet(folder + "spatialEnvSocialClustersTessSet8.pq")

# %%
ax = spatialEnvSocialTess8.plot('clusters_form_social_envPred_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected fewer) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (6)")

ax = spatialEnvSocialTess8.plot('clusters_form_social_envPred_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected fewer) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (7)")

ax = spatialEnvSocialTess8.plot('clusters_form_social_envPred_9', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected fewer) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (9)")

ax = spatialEnvSocialTess8.plot('clusters_form_social_envPred_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected fewer) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (10)")

ax = spatialEnvSocialTess8.plot('clusters_form_social_envPred_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected fewer) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (12)")

ax = spatialEnvSocialTess8.plot('clusters_form_social_envPred_13', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected fewer) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (13)")

ax = spatialEnvSocialTess8.plot('clusters_form_social_envPred_17', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected fewer) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (17)")

# %% [markdown]
# ##### Set 9 Plots

# %%
spatialEnvSocialTess9['clusters_form_social_envPred_3'] = km9_3.labels_
spatialEnvSocialTess9['clusters_form_social_envPred_4'] = km9_4.labels_
spatialEnvSocialTess9['clusters_form_social_envPred_7'] = km9_7.labels_
spatialEnvSocialTess9['clusters_form_social_envPred_9'] = km9_9.labels_
spatialEnvSocialTess9['clusters_form_social_envPred_10'] = km9_10.labels_
spatialEnvSocialTess9['clusters_form_social_envPred_12'] = km9_12.labels_
spatialEnvSocialTess9['clusters_form_social_envPred_17'] = km9_17.labels_
spatialEnvSocialTess9['clusters_form_social_envPred_20'] = km9_20.labels_

spatialEnvSocialTess9.to_parquet(folder + "spatialEnvSocialClustersTessSet9.pq")

# %%
ax = spatialEnvSocialTess9.plot('clusters_form_social_envPred_3', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (3)")

ax = spatialEnvSocialTess9.plot('clusters_form_social_envPred_4', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (4)")

ax = spatialEnvSocialTess9.plot('clusters_form_social_envPred_7', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (7)")

ax = spatialEnvSocialTess9.plot('clusters_form_social_envPred_9', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (9)")

ax = spatialEnvSocialTess9.plot('clusters_form_social_envPred_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (10)")

ax = spatialEnvSocialTess9.plot('clusters_form_social_envPred_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (12)")

ax = spatialEnvSocialTess9.plot('clusters_form_social_envPred_17', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (17)")

ax = spatialEnvSocialTess9.plot('clusters_form_social_envPred_20', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (20)")

# %% [markdown]
# ##### Set 10 Plots

# %%
spatialEnvSocialTess10['clusters_form_social_envPred_6'] = km10_6.labels_
spatialEnvSocialTess10['clusters_form_social_envPred_9'] = km10_9.labels_
spatialEnvSocialTess10['clusters_form_social_envPred_10'] = km10_10.labels_
spatialEnvSocialTess10['clusters_form_social_envPred_12'] = km10_12.labels_
spatialEnvSocialTess10['clusters_form_social_envPred_14'] = km10_14.labels_
spatialEnvSocialTess10['clusters_form_social_envPred_15'] = km10_15.labels_
spatialEnvSocialTess10['clusters_form_social_envPred_18'] = km10_18.labels_

spatialEnvSocialTess10.to_parquet(folder + "spatialEnvSocialClustersTessSet10.pq")

# %%
ax = spatialEnvSocialTess10.plot('clusters_form_social_envPred_6', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (no scoring) Clusters ETCs (6)")

ax = spatialEnvSocialTess10.plot('clusters_form_social_envPred_9', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (no scoring) Clusters ETCs (9)")

ax = spatialEnvSocialTess10.plot('clusters_form_social_envPred_10', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (no scoring) Clusters ETCs (10)")

ax = spatialEnvSocialTess10.plot('clusters_form_social_envPred_12', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (no scoring) Clusters ETCs (12)")

ax = spatialEnvSocialTess10.plot('clusters_form_social_envPred_14', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (no scoring) Clusters ETCs (14)")

ax = spatialEnvSocialTess10.plot('clusters_form_social_envPred_15', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (no scoring) Clusters ETCs (15)")

ax = spatialEnvSocialTess10.plot('clusters_form_social_envPred_18', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (no scoring) Clusters ETCs (18)")

# %% [markdown]
# ##### Final Selected Plots

# %% [markdown]
# In order to choose the clusters that made the most sense, we analyzed the plots to select which ones made the most sense qualitatively (based on the judgement of the author), then compared these selected plots with their silhouette score. The plots that made the most sesnse qualitatively and had the highest silhouette scores were then selected since this meant they were the most easily differentiated from one another in the KMeans clustering, which means the trends of each cluster would be most unique to their cluster.
# These ended up being set 8 with 12 clusters; and set 9 with 12 clusters as well. 

# %%
fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(20, 20))

spatialEnvSocialTess8.plot('clusters_form_social_envPred_12', ax=ax1, categorical=True, legend=True,# figsize=(20, 20), 
                           cmap='tab20', label="Set 8, 12 clusters")
ax1.set_axis_off()
ax1.set_title("Set 8 - 12 clusters")
# ax.set_title("Spatial form (selected fewer) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (12)")

spatialEnvSocialTess9.plot('clusters_form_social_envPred_12', ax=ax2, categorical=True, legend=True,# figsize=(20, 20), 
                           cmap='tab20', label="Set 9, 12 clusters")
ax2.set_axis_off()
ax2.set_title("Set 9 - 12 clusters")
# ax.set_title("Spatial form (selected Original) + Social Sustainability Indicators (only access) + Environmental Prediction (only scoring) Clusters ETCs (12)")


# %% [markdown]
# ##### Old data (less useful)

# %%
cg_fullTessClstr1 = Clustergram(range(1, 50), n_init=30, random_state=12345)
cg_fullTessClstr1.fit(dataTess1)
# fig, axs = plt.subplots(2, figsize=(10, 10), sharex=True)
axs1 = cg_fullTessClstr1.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
axs1.set_title("Environmnetal Prediction, Social Connectivity, and Spatial Signatures set 1 Silhouette score")

# %%
cg_fullTessClstr2 = Clustergram(range(1, 50), n_init=30, random_state=12345)
cg_fullTessClstr2.fit(dataTess2)
# fig, axs = plt.subplots(2, figsize=(10, 10), sharex=True)
axs2 = cg_fullTessClstr2.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
axs2.set_title("Environmnetal Prediction, Social Connectivity, and Spatial Signatures set 2 Silhouette score")

# %%
cg_fullTessClstr3 = Clustergram(range(1, 50), n_init=30, random_state=12345)
cg_fullTessClstr3.fit(dataTess3)
# fig, axs = plt.subplots(2, figsize=(10, 10), sharex=True)
axs3 = cg_fullTessClstr3.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
axs3.set_title("Environmnetal Prediction, Social Connectivity, and Spatial Signatures set 2 Silhouette score total set 3")

# %%
axs3 = cg_fullTessClstr3.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
axs3.set_title("Environmnetal Prediction, Social Connectivity, and Spatial Signatures set 2 Silhouette score total set 3")
axs3.axvline(8, color=c[4], ymin=.02, ymax=.98)


# %%
c = ugg.COLORS
ax = cg_fullTessClstr3.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)

# %%
cg_fullTessClstr4 = Clustergram(range(1, 50), n_init=30, random_state=12345)
cg_fullTessClstr4.fit(dataTess4)
# fig, axs = plt.subplots(2, figsize=(10, 10), sharex=True)
axs4 = cg_fullTessClstr4.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
axs4.set_title("Environmnetal Prediction, Social Connectivity, and Spatial Signatures set 2 Silhouette score total set 4")

# %%
axs4 = cg_fullTessClstr4.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
axs4.set_title("Environmnetal Prediction, Social Connectivity, and Spatial Signatures set 2 Silhouette score total set 4")
axs4.axvline(5, color=c[4], ymin=.02, ymax=.98)


# %%
c = ugg.COLORS
ax = cg_fullTessClstr4.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)

# %%
cg_fullTessClstr5 = Clustergram(range(1, 50), n_init=30, random_state=12345)
cg_fullTessClstr5.fit(dataTess5)
# fig, axs = plt.subplots(2, figsize=(10, 10), sharex=True)
axs5 = cg_fullTessClstr5.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
axs5.set_title("Environmnetal Prediction, Social Connectivity, and Spatial Signatures set 2 Silhouette score total set 5")

# %%
axs5 = cg_fullTessClstr5.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
axs5.set_title("Environmnetal Prediction, Social Connectivity, and Spatial Signatures set 2 Silhouette score total set 5")
axs5.axvline(5, color=c[4], ymin=.02, ymax=.98)


# %%
c = ugg.COLORS
ax = cg_fullTessClstr5.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)

# %%
cg_fullTessClstr6 = Clustergram(range(1, 50), n_init=30, random_state=12345)
cg_fullTessClstr6.fit(dataTess6)
# fig, axs = plt.subplots(2, figsize=(10, 10), sharex=True)
axs6 = cg_fullTessClstr6.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
axs6.set_title("Environmnetal Prediction, Social Connectivity, and Spatial Signatures set 2 Silhouette score total set 6")

# %%
axs6 = cg_fullTessClstr6.silhouette_score().plot(xlabel="Number of clusters (k)", ylabel="Silhouette score")
axs6.set_title("Environmnetal Prediction, Social Connectivity, and Spatial Signatures set 2 Silhouette score total set 6")
axs6.axvline(5, color=c[4], ymin=.02, ymax=.98)


# %%
c = ugg.COLORS
ax = cg_fullTessClstr6.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)

# %% [markdown]
# This illustrates that the only space where the clustering is reasonable according to the silhouette score is at 8 clusters. The score is still lower than ideal, however it is the best score theres been. This indicates that the dimensionality of the larger set of spatial features is perhaps too high to be able to adequately cluster them, however with the smaller spatial features it seems fairly doable to cluster.

# %%
km1_8 = MiniBatchKMeans(n_clusters=8, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1)
km1_12 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1)
km1_17 = MiniBatchKMeans(n_clusters=17, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess1)

# %%
spatialEnvSocialTess1['clusters_all1_8'] = km1_8.labels_
spatialEnvSocialTess1['clusters_all1_12'] = km1_12.labels_
spatialEnvSocialTess1['clusters_all1_17'] = km1_17.labels_
spatialEnvSocialTess1.to_parquet(folder + "spatialEnvSocialClustersTess1.pq")

# %%
inertia1_8 = km1_8.inertia_

# %%
c = ugg.COLORS
ax = cg_fullTessCluster2.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramFullTess2.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# Clusters at 10, 14, 18

# %%
cg_labels = cg_fullTessCluster2.labels
cg_labels.columns = cg_labels.columns.astype(str)
cg_labels.to_parquet(folder + 'clustergram_FullTess2_labels.pq')

# %%
km2_10 = MiniBatchKMeans(n_clusters=8, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2)
km2_14 = MiniBatchKMeans(n_clusters=12, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2)
km2_18 = MiniBatchKMeans(n_clusters=17, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess2)

# %%
spatialEnvSocialTess2['clusters_all2_8'] = km2_10.labels_
spatialEnvSocialTess2['clusters_all2_14'] = km2_14.labels_
spatialEnvSocialTess2['clusters_all2_18'] = km2_18.labels_
spatialEnvSocialTess2.to_parquet(folder + "spatialEnvSocialClustersTess2.pq")

# %%
ax = spatialEnvSocialTess2.plot('clusters_all2_8', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial set 2 + Environmental Prediction + Social Form Clusters (8 clusters)")
ax.set_axis_off()

# %%
km3 = MiniBatchKMeans(n_clusters=8, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess3)
km4 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess4)
km5 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess5)
km6 = MiniBatchKMeans(n_clusters=5, n_init=1000, random_state=12345, batch_size=100000).fit(dataTess6)


# %%
spatialEnvSocialTess3['clusters_all'] = km3.labels_
spatialEnvSocialTess4['clusters_all'] = km4.labels_
spatialEnvSocialTess5['clusters_all'] = km5.labels_
spatialEnvSocialTess6['clusters_all'] = km6.labels_
spatialEnvSocialTess3.to_parquet(folder + "spatialEnvSocialClustersTess3.pq")
spatialEnvSocialTess4.to_parquet(folder + "spatialEnvSocialClustersTess4.pq")
spatialEnvSocialTess5.to_parquet(folder + "spatialEnvSocialClustersTess5.pq")
spatialEnvSocialTess6.to_parquet(folder + "spatialEnvSocialClustersTess6.pq")

# %% [markdown]
# It seems that set 3 is the best for defining the clusters, since there is a large amount of clusters and from a qualitative analysis the diversity of the land it represents seems to be consistent with the expectations based on knowledge about this area

# %%
ax = spatialEnvSocialTess3.plot('clusters_all', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial set 2 + Environmental Prediction + Social Form Clusters set 3 (8 clusters)")
ax.set_axis_off()

ax = spatialEnvSocialTess4.plot('clusters_all', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial set 2 + Environmental Prediction + Social Form Clusters set 4 (5 clusters)")
ax.set_axis_off()

ax = spatialEnvSocialTess5.plot('clusters_all', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial set 2 + Environmental Prediction + Social Form Clusters set 5 (5 clusters)")
ax.set_axis_off()

ax = spatialEnvSocialTess6.plot('clusters_all', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_title("Spatial set 2 + Environmental Prediction + Social Form Clusters set 6 (5 clusters)")
ax.set_axis_off()

# %% [markdown]
# ##### Process Data for Bar Charts

# %%
spatialEnvSocialTess8wPop = spatialEnvSocialTess8.drop(columns=["clusters_form_social_envPred_5", 
                                                                "clusters_form_social_envPred_7", 
                                                                "clusters_form_social_envPred_10", 
                                                                "clusters_form_social_envPred_11",
                                                                "clusters_form_social_envPred_14",
                                                                "clusters_form_social_envPred_6", 
                                                                "clusters_form_social_envPred_9", 
                                                                "clusters_form_social_envPred_13", 
                                                                "clusters_form_social_envPred_17"]).merge(
                                                                    spatialEnvSocialTess2[["geometry","POP20"]], on="geometry")
spatialEnvSocialTess8wPop.rename(columns={'clusters_form_social_envPred_12': 'clusters_all'}, inplace=True)

# %%
spatialEnvSocialTess9wPop = spatialEnvSocialTess9.drop(columns=["clusters_form_social_envPred_5", 
                                                                "clusters_form_social_envPred_7", 
                                                                "clusters_form_social_envPred_10", 
                                                                "clusters_form_social_envPred_11",
                                                                "clusters_form_social_envPred_14",
                                                                "clusters_form_social_envPred_3", 
                                                                "clusters_form_social_envPred_4", 
                                                                "clusters_form_social_envPred_9", 
                                                                "clusters_form_social_envPred_17", 
                                                                "clusters_form_social_envPred_20"]).merge(
                                                                    spatialEnvSocialTess2[["geometry","POP20"]], on="geometry")
spatialEnvSocialTess9wPop.rename(columns={'clusters_form_social_envPred_12': 'clusters_all'}, inplace=True)

# %% [markdown]
# One issue with the spatial data is that in order to get a decent idea of the trends in a localized area, a convolution was done with the interquartile ranges reported. This should provide more relevant area about contextual character however it could also be a limitation.

# %%
spatialEnvSocialTess9wPop.sscCCo_q2.max()

# %%
spatialEnvSocialTess3wPop = spatialEnvSocialTess3.merge(spatialEnvSocialTess2[["geometry","POP20"]], on="geometry")

# %%
def meanByPop(df, pop, maxPop, minPop, nonNorm):
    cols = df.columns
    dfMean = gpd.GeoDataFrame()
    # maxPop = np.max(df[pop])
    # minPop = np.min(df[pop])
    popNormDF = (df[pop] - minPop)/(maxPop-minPop)

    for col in cols:
        if col == pop:  
            dfMean[col] = df[col].sum()
            continue
        elif col in nonNorm:
            dfMean[col] = df[col]
            continue
        dfMean[col] = df[col]*popNormDF
        dfMean[col] = dfMean[col].mean()
    
    return dfMean

# %%
def meanForDF(df, pop, nonNorm):
    cols = df.columns
    dfMean = gpd.GeoDataFrame()

    for col in cols:
        if col == pop:  
            dfMean[col] = df[col].sum()
            continue
        elif col in nonNorm:
            dfMean[col] = df[col]
            continue
        dfMean[col] = df[col]
        dfMean[col] = dfMean[col].mean()

    return dfMean

# %%
clusters = spatialEnvSocialTess8wPop.clusters_all.unique()

# %%
nonNorm = ["geometry", "clusters_all"]
pop = "POP20"

# %%
spatialEnvSocialTess8Mean = gpd.GeoDataFrame()
spatialEnvSocialTess9Mean = gpd.GeoDataFrame()
spatialEnvSocialTess8MeanDirect = gpd.GeoDataFrame()
spatialEnvSocialTess9MeanDirect = gpd.GeoDataFrame()

maxPop8 = np.max(spatialEnvSocialTess8wPop[pop])
maxPop9 = np.max(spatialEnvSocialTess9wPop[pop])
minPop8 = np.min(spatialEnvSocialTess8wPop[pop])
minPop9 = np.min(spatialEnvSocialTess9wPop[pop])

# Set 8 and Set 9 have the same number of clusters
for clust in clusters:
    # Set 8
    spatEnvSocTess8 = spatialEnvSocialTess8wPop.loc[spatialEnvSocialTess8wPop.clusters_all == clust]
    spatEnvSocTess8MeanTemp =  meanByPop(spatEnvSocTess8, pop, maxPop8, minPop8, nonNorm)
    spatialEnvSocialTess8Mean = pd.concat([spatialEnvSocialTess8Mean, spatEnvSocTess8MeanTemp])
    spatEnvSocTess8MeanTemp =  meanForDF(spatEnvSocTess8, pop, nonNorm)
    spatialEnvSocialTess8MeanDirect = pd.concat([spatialEnvSocialTess8MeanDirect, spatEnvSocTess8MeanTemp])

    # Set 9
    spatEnvSocTess9 = spatialEnvSocialTess9wPop.loc[spatialEnvSocialTess9wPop.clusters_all == clust]
    spatEnvSocTess9MeanTemp =  meanByPop(spatEnvSocTess9, pop, maxPop9, minPop9, nonNorm)
    spatialEnvSocialTess9Mean = pd.concat([spatialEnvSocialTess9Mean, spatEnvSocTess9MeanTemp])
    spatEnvSocTess9MeanTemp =  meanForDF(spatEnvSocTess9, pop, nonNorm)
    spatialEnvSocialTess9MeanDirect = pd.concat([spatialEnvSocialTess9MeanDirect, spatEnvSocTess9MeanTemp])
# spatialEnvSocialTess3Mean = spatialEnvSocialTess3Mean.T

# %%
spatialEnvSocialTess8Mean.POP20.max()

# %%
spatialEnvSocialTess8Mean["clusters"] = spatialEnvSocialTess8Mean["clusters_all"]
spatialEnvSocialTess8MeanDirect["clusters"] = spatialEnvSocialTess8MeanDirect["clusters_all"]
spatialEnvSocialTess9Mean["clusters"] = spatialEnvSocialTess9Mean["clusters_all"]
spatialEnvSocialTess9MeanDirect["clusters"] = spatialEnvSocialTess9MeanDirect["clusters_all"]

# %%
finalSet8 = spatialEnvSocialTess8Mean.dissolve('clusters', aggfunc="mean")
finalSet8_v2 = spatialEnvSocialTess8MeanDirect.dissolve('clusters', aggfunc="mean")
finalSet9 = spatialEnvSocialTess9Mean.dissolve('clusters', aggfunc="mean")
finalSet9_v2 = spatialEnvSocialTess9MeanDirect.dissolve('clusters', aggfunc="mean")


# %%
finalSet8.head(1)

# %%
finalSet8_v2.head(1)

# %%
finalSetEx = finalSet9_v2.copy()
finalSetEx["DataBruh"] = 1

# %%
numRows = iter(range(len(finalSetEx)))

# %%
j = 5
for i in numRows:
    finalSetEx["DataBruh"][i] = j
    j = j + 1

# %%
finalSetEx

# %%
ax = finalSet8.plot('clusters_all', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()

ax = finalSet9.plot('clusters_all', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()

# %%
finalSet8.columns

# %%
finalSet9.columns

# %%
used_characteristics_dict_v2 = {
    'sscCCo': 'circular compactness of ETC', #
    'sicCAR': 'covered area ratio of ETC', # building cover
    'lddNDe': 'local node density of street network',
    'sdsSPO': 'openness of street profile', #
    'ltbIBD': 'mean inter-building distance',
    'lieWCe': 'area-weighted ETCs of enclosure', # 
}

# %%
spatialEnvSocialTess9wPop.head(1)

# %%
circCompactHeatCorr = spatialEnvSocialTess8wPop[["sscCCo_q2", "heat_score"]].corr(method="kendall")
buildCoverHeatCorr = spatialEnvSocialTess8wPop[["sicCAR_q2", "heat_score"]].corr(method="kendall")
streetOpenHeatCorr = spatialEnvSocialTess8wPop[["sdsSPO_q2", "heat_score"]].corr(method="kendall")
nodeDensHeatCorr = spatialEnvSocialTess8wPop[["lddNDe_q2", "heat_score"]].corr(method="kendall")

circCompactWaterCorr = spatialEnvSocialTess8wPop[["sscCCo_q2", "water_score"]].corr(method="kendall")
buildCoverWaterCorr = spatialEnvSocialTess8wPop[["sicCAR_q2", "water_score"]].corr(method="kendall")
streetOpenWaterCorr = spatialEnvSocialTess8wPop[["sdsSPO_q2", "water_score"]].corr(method="kendall")
nodeDensWaterCorr = spatialEnvSocialTess8wPop[["lddNDe_q2", "water_score"]].corr(method="kendall")

circCompactAccessCorr = spatialEnvSocialTess8wPop[["sscCCo_q2", "accessScore"]].corr(method="kendall")
buildCoverAccessCorr = spatialEnvSocialTess8wPop[["sicCAR_q2", "accessScore"]].corr(method="kendall")
streetOpenAccessCorr = spatialEnvSocialTess8wPop[["sdsSPO_q2", "accessScore"]].corr(method="kendall")
nodeDensAccessCorr = spatialEnvSocialTess8wPop[["lddNDe_q2", "accessScore"]].corr(method="kendall")

correlationHeatSet8 = pd.DataFrame([circCompactHeatCorr["heat_score"]["sscCCo_q2"],buildCoverHeatCorr["heat_score"]["sicCAR_q2"],
                                 streetOpenHeatCorr["heat_score"]["sdsSPO_q2"],nodeDensHeatCorr["heat_score"]["lddNDe_q2"]]).T
correlationWaterSet8 = pd.DataFrame([circCompactWaterCorr["water_score"]["sscCCo_q2"],buildCoverWaterCorr["water_score"]["sicCAR_q2"],
                                 streetOpenWaterCorr["water_score"]["sdsSPO_q2"],nodeDensWaterCorr["water_score"]["lddNDe_q2"]]).T
correlationAccessSet8 = pd.DataFrame([circCompactAccessCorr["accessScore"]["sscCCo_q2"],buildCoverAccessCorr["accessScore"]["sicCAR_q2"],
                                 streetOpenAccessCorr["accessScore"]["sdsSPO_q2"],nodeDensAccessCorr["accessScore"]["lddNDe_q2"]]).T

correlationHeatSet8.columns = ["sscCCo", "sicCAR","sdsSPO", "lddNDe"]
correlationHeatSet8.index = ["heat_score"]
correlationWaterSet8.columns = ["sscCCo", "sicCAR","sdsSPO", "lddNDe"]
correlationWaterSet8.index = ["water_score"]
correlationAccessSet8.columns = ["sscCCo", "sicCAR","sdsSPO", "lddNDe"]
correlationAccessSet8.index = ["access_score"]
correlationSet8 = pd.concat([correlationHeatSet8,correlationWaterSet8,correlationAccessSet8])



# %%
correlationSet8

# %%
circCompactHeatCorr = spatialEnvSocialTess9wPop[["sscCCo_q2", "heat_score"]].corr(method="kendall")
buildCoverHeatCorr = spatialEnvSocialTess9wPop[["sicCAR_q2", "heat_score"]].corr(method="kendall")
streetOpenHeatCorr = spatialEnvSocialTess9wPop[["lieWCe_q2", "heat_score"]].corr(method="kendall")
nodeDensHeatCorr = spatialEnvSocialTess9wPop[["lddNDe_q2", "heat_score"]].corr(method="kendall")

circCompactWaterCorr = spatialEnvSocialTess9wPop[["sscCCo_q2", "water_score"]].corr(method="kendall")
buildCoverWaterCorr = spatialEnvSocialTess9wPop[["sicCAR_q2", "water_score"]].corr(method="kendall")
streetOpenWaterCorr = spatialEnvSocialTess9wPop[["lieWCe_q2", "water_score"]].corr(method="kendall")
nodeDensWaterCorr = spatialEnvSocialTess9wPop[["lddNDe_q2", "water_score"]].corr(method="kendall")

circCompactAccessCorr = spatialEnvSocialTess9wPop[["sscCCo_q2", "accessScore"]].corr(method="kendall")
buildCoverAccessCorr = spatialEnvSocialTess9wPop[["sicCAR_q2", "accessScore"]].corr(method="kendall")
streetOpenAccessCorr = spatialEnvSocialTess9wPop[["lieWCe_q2", "accessScore"]].corr(method="kendall")
nodeDensAccessCorr = spatialEnvSocialTess9wPop[["lddNDe_q2", "accessScore"]].corr(method="kendall")

correlationHeatSet9 = pd.DataFrame([circCompactHeatCorr["heat_score"]["sscCCo_q2"],buildCoverHeatCorr["heat_score"]["sicCAR_q2"],
                                 streetOpenHeatCorr["heat_score"]["lieWCe_q2"],nodeDensHeatCorr["heat_score"]["lddNDe_q2"]]).T
correlationWaterSet9 = pd.DataFrame([circCompactWaterCorr["water_score"]["sscCCo_q2"],buildCoverWaterCorr["water_score"]["sicCAR_q2"],
                                 streetOpenWaterCorr["water_score"]["lieWCe_q2"],nodeDensWaterCorr["water_score"]["lddNDe_q2"]]).T
correlationAccessSet9 = pd.DataFrame([circCompactAccessCorr["accessScore"]["sscCCo_q2"],buildCoverAccessCorr["accessScore"]["sicCAR_q2"],
                                 streetOpenAccessCorr["accessScore"]["lieWCe_q2"],nodeDensAccessCorr["accessScore"]["lddNDe_q2"]]).T

correlationHeatSet9.columns = ["sscCCo", "sicCAR","lieWCe_q2", "lddNDe"]
correlationHeatSet9.index = ["heat_score"]
correlationWaterSet9.columns = ["sscCCo", "sicCAR","lieWCe_q2", "lddNDe"]
correlationWaterSet9.index = ["water_score"]
correlationAccessSet9.columns = ["sscCCo", "sicCAR","lieWCe_q2", "lddNDe"]
correlationAccessSet9.index = ["access_score"]
correlationSet9 = pd.concat([correlationHeatSet9,correlationWaterSet9,correlationAccessSet9])


# %%
correlationSet9

# %% [markdown]
# ### Bar Charts for Spatial Signatures

# %% [markdown]
# ##### Bar Charts for Form Set 8

# %%
circularCompactnessBarSet8 = finalSet8[["sscCCo_q1","sscCCo_q2","sscCCo_q3"]].plot.bar(rot=0)
circularCompactnessBarSet8.legend(['First Quartile',
                         'Second Quartile',
                         'Third Quartile'])
circularCompactnessBarSet8.set_ylabel(used_characteristics_dict_v2["sscCCo"])

circularCompactnessMedianBarSet8 = finalSet8[["sscCCo_q2"]].plot.bar(rot=0)
circularCompactnessMedianBarSet8.legend(['Median'])
circularCompactnessMedianBarSet8.set_ylabel(used_characteristics_dict_v2["sscCCo"])

coveredAreaRatioBarSet8 = finalSet8[["sicCAR_q1","sicCAR_q2","sicCAR_q3"]].plot.bar(rot=0)
coveredAreaRatioBarSet8.legend(['First Quartile',
                         'Second Quartile',
                         'Third Quartile'])
coveredAreaRatioBarSet8.set_ylabel(used_characteristics_dict_v2["sicCAR"])

coveredAreaRatioMedianBarSet8 = finalSet8[["sicCAR_q2"]].plot.bar(rot=0)
coveredAreaRatioMedianBarSet8.legend(['Median'])
coveredAreaRatioMedianBarSet8.set_ylabel(used_characteristics_dict_v2["sicCAR"])

streetProfileOpennessBarSet8 = finalSet8[["sdsSPO_q1","sdsSPO_q2","sdsSPO_q3"]].plot.bar(rot=0)#x='clusters_all', y='lddNDe_q1', rot=0)
streetProfileOpennessBarSet8.legend(['First Quartile',
                         'Second Quartile',
                         'Third Quartile'])
streetProfileOpennessBarSet8.set_ylabel(used_characteristics_dict_v2["sdsSPO"])

streetProfileOpennessMedianBarSet8 = finalSet8[["sdsSPO_q2"]].plot.bar(rot=0)#x='clusters_all', y='lddNDe_q1', rot=0)
streetProfileOpennessMedianBarSet8.legend(['Median'])
streetProfileOpennessMedianBarSet8.set_ylabel(used_characteristics_dict_v2["sdsSPO"])

nodeDensitySNBarSet8 = finalSet8[["lddNDe_q1","lddNDe_q2","lddNDe_q3"]].plot.bar(rot=0)#x='clusters_all', y='lddNDe_q1', rot=0)
nodeDensitySNBarSet8.legend(['First Quartile',
                         'Second Quartile',
                         'Third Quartile'])
nodeDensitySNBarSet8.set_ylabel(used_characteristics_dict_v2["lddNDe"])

nodeDensitySNMedianBarSet8 = finalSet8[["lddNDe_q2"]].plot.bar(rot=0)#x='clusters_all', y='lddNDe_q1', rot=0)
nodeDensitySNMedianBarSet8.legend(['Median'])
nodeDensitySNMedianBarSet8.set_ylabel(used_characteristics_dict_v2["lddNDe"])


# %%
circularCompactnessBarSet8_v2 = finalSet8_v2[["sscCCo_q1","sscCCo_q2","sscCCo_q3"]].plot.bar(rot=0)
circularCompactnessBarSet8_v2.legend(['First Quartile',
                         'Second Quartile',
                         'Third Quartile'])
circularCompactnessBarSet8_v2.set_ylabel(used_characteristics_dict_v2["sscCCo"])

circularCompactnessMedianBarSet8_v2 = finalSet8_v2[["sscCCo_q2"]].plot.bar(rot=0)
circularCompactnessMedianBarSet8_v2.legend(['Median'])
circularCompactnessMedianBarSet8_v2.set_ylabel(used_characteristics_dict_v2["sscCCo"])

coveredAreaRatioBarSet8_v2 = finalSet8_v2[["sicCAR_q1","sicCAR_q2","sicCAR_q3"]].plot.bar(rot=0)
coveredAreaRatioBarSet8_v2.legend(['First Quartile',
                         'Second Quartile',
                         'Third Quartile'])
coveredAreaRatioBarSet8_v2.set_ylabel(used_characteristics_dict_v2["sicCAR"])

coveredAreaRatioMedianBarSet8_v2 = finalSet8_v2[["sicCAR_q2"]].plot.bar(rot=0)
coveredAreaRatioMedianBarSet8_v2.legend(['Median'])
coveredAreaRatioMedianBarSet8_v2.set_ylabel(used_characteristics_dict_v2["sicCAR"])

streetProfileOpennessBarSet8_v2 = finalSet8_v2[["sdsSPO_q1","sdsSPO_q2","sdsSPO_q3"]].plot.bar(rot=0)#x='clusters_all', y='lddNDe_q1', rot=0)
streetProfileOpennessBarSet8_v2.legend(['First Quartile',
                         'Second Quartile',
                         'Third Quartile'])
streetProfileOpennessBarSet8_v2.set_ylabel(used_characteristics_dict_v2["sdsSPO"])

streetProfileOpennessMedianBarSet8_v2 = finalSet8_v2[["sdsSPO_q2"]].plot.bar(rot=0)#x='clusters_all', y='lddNDe_q1', rot=0)
streetProfileOpennessMedianBarSet8_v2.legend(['Median'])
streetProfileOpennessMedianBarSet8_v2.set_ylabel(used_characteristics_dict_v2["sdsSPO"])

nodeDensitySNBarSet8_v2 = finalSet8_v2[["lddNDe_q1","lddNDe_q2","lddNDe_q3"]].plot.bar(rot=0)#x='clusters_all', y='lddNDe_q1', rot=0)
nodeDensitySNBarSet8_v2.legend(['First Quartile',
                         'Second Quartile',
                         'Third Quartile'])
nodeDensitySNBarSet8_v2.set_ylabel(used_characteristics_dict_v2["lddNDe"])

nodeDensitySNMedianBarSet8_v2 = finalSet8_v2[["lddNDe_q2"]].plot.bar(rot=0)#x='clusters_all', y='lddNDe_q1', rot=0)
nodeDensitySNMedianBarSet8_v2.legend(['Median'])
nodeDensitySNMedianBarSet8_v2.set_ylabel(used_characteristics_dict_v2["lddNDe"])


# %% [markdown]
# Clearly scaling based on population causes the clusters with the higher population to have higher values - which probably does not make sense for the analysis.

# %% [markdown]
# ##### Bar Charts for Form Set 9

# %%
circularCompactnessBarSet9 = finalSet9[["sscCCo_q1","sscCCo_q2","sscCCo_q3"]].plot.bar(rot=0)
circularCompactnessBarSet9.legend(['First Quartile',
                         'Second Quartile',
                         'Third Quartile'])
circularCompactnessBarSet9.set_ylabel(used_characteristics_dict_v2["sscCCo"])

circularCompactnessMedianBarSet9 = finalSet9[["sscCCo_q2"]].plot.bar(rot=0)
circularCompactnessMedianBarSet9.legend(['Median'])
circularCompactnessMedianBarSet9.set_ylabel(used_characteristics_dict_v2["sscCCo"])

coveredAreaRatioBarSet9 = finalSet9[["sicCAR_q1","sicCAR_q2","sicCAR_q3"]].plot.bar(rot=0)
coveredAreaRatioBarSet9.legend(['First Quartile',
                         'Second Quartile',
                         'Third Quartile'])
coveredAreaRatioBarSet9.set_ylabel(used_characteristics_dict_v2["sicCAR"])

coveredAreaRatioMedianBarSet9 = finalSet9[["sicCAR_q2"]].plot.bar(rot=0)
coveredAreaRatioMedianBarSet9.legend(['Median'])
coveredAreaRatioMedianBarSet9.set_ylabel(used_characteristics_dict_v2["sicCAR"])

nodeDensitySNBarSet9 = finalSet9[["lddNDe_q1","lddNDe_q2","lddNDe_q3"]].plot.bar(rot=0)#x='clusters_all', y='lddNDe_q1', rot=0)
nodeDensitySNBarSet9.legend(['First Quartile',
                         'Second Quartile',
                         'Third Quartile'])
nodeDensitySNBarSet9.set_ylabel(used_characteristics_dict_v2["lddNDe"])

nodeDensitySNMedianBarSet9 = finalSet9[["lddNDe_q2"]].plot.bar(rot=0)#x='clusters_all', y='lddNDe_q1', rot=0)
nodeDensitySNMedianBarSet9.legend(['Median'])
nodeDensitySNMedianBarSet9.set_ylabel(used_characteristics_dict_v2["lddNDe"])

areaETCEnclosureBarSet9 = finalSet9[["lieWCe_q1","lieWCe_q2","lieWCe_q3"]].plot.bar(rot=0)#x='clusters_all', y='lddNDe_q1', rot=0)
areaETCEnclosureBarSet9.legend(['First Quartile',
                         'Second Quartile',
                         'Third Quartile'])
areaETCEnclosureBarSet9.set_ylabel(used_characteristics_dict_v2["lieWCe"])

areaETCEnclosureMedianBarSet9 = finalSet9[["lieWCe_q2"]].plot.bar(rot=0)#x='clusters_all', y='lddNDe_q1', rot=0)
areaETCEnclosureMedianBarSet9.legend(['Median'])
areaETCEnclosureMedianBarSet9.set_ylabel(used_characteristics_dict_v2["lieWCe"])

# %%
circularCompactnessBarSet9_v2 = finalSet9_v2[["sscCCo_q1","sscCCo_q2","sscCCo_q3"]].plot.bar(rot=0)
circularCompactnessBarSet9_v2.legend(['First Quartile',
                         'Second Quartile',
                         'Third Quartile'])
circularCompactnessBarSet9_v2.set_ylabel(used_characteristics_dict_v2["sscCCo"])

circularCompactnessMedianBarSet9_v2 = finalSet9_v2[["sscCCo_q2"]].plot.bar(rot=0)
circularCompactnessMedianBarSet9_v2.legend(['Median'])
circularCompactnessMedianBarSet9_v2.set_ylabel(used_characteristics_dict_v2["sscCCo"])

coveredAreaRatioBarSet9_v2 = finalSet9_v2[["sicCAR_q1","sicCAR_q2","sicCAR_q3"]].plot.bar(rot=0)
coveredAreaRatioBarSet9_v2.legend(['First Quartile',
                         'Second Quartile',
                         'Third Quartile'])
coveredAreaRatioBarSet9_v2.set_ylabel(used_characteristics_dict_v2["sicCAR"])

coveredAreaRatioMedianBarSet9_v2 = finalSet9_v2[["sicCAR_q2"]].plot.bar(rot=0)
coveredAreaRatioMedianBarSet9_v2.legend(['Median'])
coveredAreaRatioMedianBarSet9_v2.set_ylabel(used_characteristics_dict_v2["sicCAR"])

nodeDensitySNBarSet9_v2 = finalSet9_v2[["lddNDe_q1","lddNDe_q2","lddNDe_q3"]].plot.bar(rot=0)#x='clusters_all', y='lddNDe_q1', rot=0)
nodeDensitySNBarSet9_v2.legend(['First Quartile',
                         'Second Quartile',
                         'Third Quartile'])
nodeDensitySNBarSet9_v2.set_ylabel(used_characteristics_dict_v2["lddNDe"])

nodeDensitySNMedianBarSet9_v2 = finalSet9_v2[["lddNDe_q2"]].plot.bar(rot=0)#x='clusters_all', y='lddNDe_q1', rot=0)
nodeDensitySNMedianBarSet9_v2.legend(['Median'])
nodeDensitySNMedianBarSet9_v2.set_ylabel(used_characteristics_dict_v2["lddNDe"])

areaETCEnclosureBarSet9_v2 = finalSet9_v2[["lieWCe_q1","lieWCe_q2","lieWCe_q3"]].plot.bar(rot=0)#x='clusters_all', y='lddNDe_q1', rot=0)
areaETCEnclosureBarSet9_v2.legend(['First Quartile',
                         'Second Quartile',
                         'Third Quartile'])
areaETCEnclosureBarSet9_v2.set_ylabel(used_characteristics_dict_v2["lieWCe"])

areaETCEnclosureMedianBarSet9_v2 = finalSet9_v2[["lieWCe_q2"]].plot.bar(rot=0)#x='clusters_all', y='lddNDe_q1', rot=0)
areaETCEnclosureMedianBarSet9_v2.legend(['Median'])
areaETCEnclosureMedianBarSet9_v2.set_ylabel(used_characteristics_dict_v2["lieWCe"])

# %% [markdown]
# ##### Bar Charts for Environmental Prediction Scoring and Accessibility Scoring

# %% [markdown]
# From the previous section it is clear that scoring based on population normalization does not work very well, thus the data from v2 will be used.

# %%
AllScoresBarSet8 = finalSet8_v2[["heat_score","water_score","accessScore"]].plot.bar(rot=0)
AllScoresBarSet8.legend(['Heat Score',
                         'Water Score',
                         'Accessibility Score'])
AllScoresBarSet8.set_ylabel("Index Scoring")

EnvScoresBarSet8 = finalSet8_v2[["heat_score","water_score"]].plot.bar(rot=0)
EnvScoresBarSet8.legend(['Heat Score',
                         'Water Score'])
EnvScoresBarSet8.set_ylabel("Environmental Index Scoring")

HeatScoresBarSet8 = finalSet8_v2[["heat_score"]].plot.bar(rot=0)
HeatScoresBarSet8.legend(['Heat Score'])
HeatScoresBarSet8.set_ylabel("Heat Index Scoring")

WaterScoresBarSet8 = finalSet8_v2[["water_score"]].plot.bar(rot=0)
WaterScoresBarSet8.legend(['Water Score'])
WaterScoresBarSet8.set_ylabel("Water Index Scoring")

AccessScoresBarSet8 = finalSet8_v2[["accessScore"]].plot.bar(rot=0)
AccessScoresBarSet8.legend(['Accessibility Score'])
AccessScoresBarSet8.set_ylabel("Social Sustainability Index Scoring")

PopulationBarSet8 = finalSet8_v2[["POP20"]].plot.bar(rot=0)
PopulationBarSet8.legend(['Population per cluster'])
PopulationBarSet8.set_ylabel("Total Population")

# %%
AllScoresBarSet9 = finalSet9_v2[["heat_score","water_score","accessScore"]].plot.bar(rot=0)
AllScoresBarSet9.legend(['Heat Score',
                         'Water Score',
                         'Accessibility Score'])
AllScoresBarSet9.set_ylabel("Index Scoring")

EnvScoresBarSet9 = finalSet9_v2[["heat_score","water_score"]].plot.bar(rot=0)
EnvScoresBarSet9.legend(['Heat Score',
                         'Water Score'])
EnvScoresBarSet9.set_ylabel("Environmental Index Scoring")

HeatScoresBarSet9 = finalSet9_v2[["heat_score"]].plot.bar(rot=0)
HeatScoresBarSet9.legend(['Heat Score'])
HeatScoresBarSet9.set_ylabel("Heat Index Scoring")

WaterScoresBarSet9 = finalSet9_v2[["water_score"]].plot.bar(rot=0)
WaterScoresBarSet9.legend(['Water Score'])
WaterScoresBarSet9.set_ylabel("Water Index Scoring")

AccessScoresBarSet9 = finalSet9_v2[["accessScore"]].plot.bar(rot=0)
AccessScoresBarSet9.legend(['Accessibility Score'])
AccessScoresBarSet9.set_ylabel("Social Sustainability Index Scoring")

PopulationBarSet9 = finalSet9_v2[["POP20"]].plot.bar(rot=0)
PopulationBarSet9.legend(['Population per cluster'])
PopulationBarSet9.set_ylabel("Total Population")

# %%
final.to_parquet(folder + "finalClustering.pq")

# %% [markdown]
# ##### Bar Chart comparing form and function

# %%
def normOut(df, pop, nonNorm):
    cols = df.columns
    dfNorm = gpd.GeoDataFrame()

    for col in cols:
        if col == pop or col in nonNorm:
            dfNorm[col] = df[col]
            continue
        maxV = np.max(df[col])
        minV = np.min(df[col])
        dfNorm[col] = (df[col] - minV)/(maxV - minV)

    return dfNorm


# %%
finalSet8Norm = normOut(finalSet8_v2, pop, nonNorm)
finalSet9Norm = normOut(finalSet9_v2, pop, nonNorm)

# %%
Set8BarCharCompare = finalSet8Norm[["sscCCo_q2","sicCAR_q2","sdsSPO_q2", "lddNDe_q2","heat_score",
                                    "water_score","accessScore"]].plot.bar(rot=0, figsize=(20,20))
Set8BarCharCompare.legend(['Median ' + used_characteristics_dict_v2["sscCCo"],
                         'Median ' + used_characteristics_dict_v2["sicCAR"],
                         'Median ' + used_characteristics_dict_v2["sdsSPO"],
                         'Median ' + used_characteristics_dict_v2["lddNDe"],
                         'Heat Score',
                         'Water Score',
                         'Accessibility Score'])



# %%
finalSet9Norm.columns

# %%
Set9BarCharCompare = finalSet9Norm[["sscCCo_q2","sicCAR_q2", "lddNDe_q2", "lieWCe_q2","heat_score",
                                    "water_score","accessScore"]].plot.bar(rot=0, figsize=(20,20))
Set9BarCharCompare.legend(['Median ' + used_characteristics_dict_v2["sscCCo"],
                         'Median ' + used_characteristics_dict_v2["sicCAR"],
                         'Median ' + used_characteristics_dict_v2["lddNDe"],
                         'Median ' + used_characteristics_dict_v2["lieWCe"],
                         'Heat Score',
                         'Water Score',
                         'Accessibility Score'])

# %%
finalSet9Norm[["sscCCo_q2","sicCAR_q2", "lddNDe_q2", "lieWCe_q2","heat_score",
                                    "water_score","accessScore"]].iloc[[0]].plot.bar(rot=0, figsize=(8,8))


# %%
finalSet9Norm.head(3)

# %%
fig, ax = plt.subplots(nrows=4, ncols=3, sharey=True) # create the axes

finalSet9Norm[["sscCCo_q2","sicCAR_q2", "lddNDe_q2", "lieWCe_q2","heat_score",
                                    "water_score","accessScore"]].iloc[[0]].plot.bar(rot=0, figsize=(8,8), ax=ax[0,0], legend=False)
finalSet9Norm[["sscCCo_q2","sicCAR_q2", "lddNDe_q2", "lieWCe_q2","heat_score",
                                    "water_score","accessScore"]].iloc[[1]].plot.bar(rot=0, figsize=(8,8), ax=ax[0,1], legend=False)
finalSet9Norm[["sscCCo_q2","sicCAR_q2", "lddNDe_q2", "lieWCe_q2","heat_score",
                                    "water_score","accessScore"]].iloc[[2]].plot.bar(rot=0, figsize=(8,8), ax=ax[0,2], legend=False)
finalSet9Norm[["sscCCo_q2","sicCAR_q2", "lddNDe_q2", "lieWCe_q2","heat_score",
                                    "water_score","accessScore"]].iloc[[3]].plot.bar(rot=0, figsize=(8,8), ax=ax[1,0], legend=False)
finalSet9Norm[["sscCCo_q2","sicCAR_q2", "lddNDe_q2", "lieWCe_q2","heat_score",
                                    "water_score","accessScore"]].iloc[[4]].plot.bar(rot=0, figsize=(8,8), ax=ax[1,1], legend=False)
finalSet9Norm[["sscCCo_q2","sicCAR_q2", "lddNDe_q2", "lieWCe_q2","heat_score",
                                    "water_score","accessScore"]].iloc[[5]].plot.bar(rot=0, figsize=(8,8), ax=ax[1,2], legend=False)
finalSet9Norm[["sscCCo_q2","sicCAR_q2", "lddNDe_q2", "lieWCe_q2","heat_score",
                                    "water_score","accessScore"]].iloc[[6]].plot.bar(rot=0, figsize=(8,8), ax=ax[2,0], legend=False)
finalSet9Norm[["sscCCo_q2","sicCAR_q2", "lddNDe_q2", "lieWCe_q2","heat_score",
                                    "water_score","accessScore"]].iloc[[7]].plot.bar(rot=0, figsize=(8,8), ax=ax[2,1], legend=False)
finalSet9Norm[["sscCCo_q2","sicCAR_q2", "lddNDe_q2", "lieWCe_q2","heat_score",
                                    "water_score","accessScore"]].iloc[[8]].plot.bar(rot=0, figsize=(8,8), ax=ax[2,2], legend=False)
finalSet9Norm[["sscCCo_q2","sicCAR_q2", "lddNDe_q2", "lieWCe_q2","heat_score",
                                    "water_score","accessScore"]].iloc[[9]].plot.bar(rot=0, figsize=(8,8), ax=ax[3,0], legend=False)
finalSet9Norm[["sscCCo_q2","sicCAR_q2", "lddNDe_q2", "lieWCe_q2","heat_score",
                                    "water_score","accessScore"]].iloc[[10]].plot.bar(rot=0, figsize=(8,8), ax=ax[3,1], legend=False)
finalSet9Norm[["sscCCo_q2","sicCAR_q2", "lddNDe_q2", "lieWCe_q2","heat_score",
                                    "water_score","accessScore"]].iloc[[11]].plot.bar(rot=0, figsize=(8,8), ax=ax[3,2], legend=False)
fig.legend(['Median ' + used_characteristics_dict_v2["sscCCo"],
                         'Median ' + used_characteristics_dict_v2["sicCAR"],
                         'Median ' + used_characteristics_dict_v2["lddNDe"],
                         'Median ' + used_characteristics_dict_v2["lieWCe"],
                         'Heat Score',
                         'Water Score',
                         'Accessibility Score'])

# %% [markdown]
# ## Less Useful Data after this

# %%
scaler = preprocessing.StandardScaler()
dataGrid1 = scaler.fit_transform(spatialEnvSocialGrid1.drop(columns=["geometry"]))
dataGrid1[np.isnan(dataGrid1)] = 0
scaler = preprocessing.StandardScaler()
dataGrid2 = scaler.fit_transform(spatialEnvSocialGrid2.drop(columns=["geometry"]))
dataGrid2[np.isnan(dataGrid2)] = 0

# %%
cg_fullCluster1 = Clustergram(range(1, 30), n_init=30, random_state=12345)
cg_fullCluster1.fit(dataGrid1)
cg_fullCluster2 = Clustergram(range(1, 30), n_init=30, random_state=12345)
cg_fullCluster2.fit(dataGrid2)

# %%
c = ugg.COLORS
ax = cg_fullCluster1.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramFull1.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# Number of clusters looks like it could be 11, 15, 18

# %%
cg_labels = cg_fullCluster1.labels
cg_labels.columns = cg_labels.columns.astype(str)
cg_labels.to_parquet(folder + 'clustergram_Full1_labels.pq')

# %%
km1_11 = MiniBatchKMeans(n_clusters=11, n_init=1000, random_state=12345, batch_size=100000).fit(dataGrid1)
km1_15 = MiniBatchKMeans(n_clusters=15, n_init=1000, random_state=12345, batch_size=100000).fit(dataGrid1)
km1_18 = MiniBatchKMeans(n_clusters=18, n_init=1000, random_state=12345, batch_size=100000).fit(dataGrid1)

# %%
spatialEnvSocialGrid1['clusters_all1_11'] = km1_11.labels_
spatialEnvSocialGrid1['clusters_all1_15'] = km1_15.labels_
spatialEnvSocialGrid1['clusters_all1_18'] = km1_18.labels_
spatialEnvSocialGrid1.to_parquet(folder + "spatialEnvSocialClusters1.pq")

# %%
ax = spatialEnvSocialGrid1.plot('clusters_all1_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction, Social Connectivity, and Spatial Signatures set 1 Clusters Grid (11)")
ax = spatialEnvSocialGrid1.plot('clusters_all1_15', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction, Social Connectivity, and Spatial Signatures set 1 Clusters Grid (15)")
ax = spatialEnvSocialGrid1.plot('clusters_all1_18', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction, Social Connectivity, and Spatial Signatures set 1 Clusters Grid (18)")

# %%
c = ugg.COLORS
ax = cg_fullCluster2.plot(
    figsize=(12, 12),
    linewidth=0.5,
    cluster_style={"edgecolor": "w", "color": c[2], "linewidth": 1.5},
    size=1,
    line_style={"color":c[1]},
    pca_kwargs=dict(random_state=12345),
)
# ax.set_ylim(-10, 25)
# ax.axvline(12, color=c[4], ymin=.02, ymax=.98)
plt.savefig(folder + "cgramFull2.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# Number of clusters looks like it could be 9, 16, 22

# %%
cg_labels = cg_fullCluster2.labels
cg_labels.columns = cg_labels.columns.astype(str)
cg_labels.to_parquet(folder + 'clustergram_Full2_labels.pq')

# %% [markdown]
# The optimal number in this case seems to be 9 clusters. We run final clustering with 1000 initiazations and fixed random state for reproducibility.

# %%
km2_9 = MiniBatchKMeans(n_clusters=9, n_init=1000, random_state=12345, batch_size=100000).fit(dataGrid2)
km2_16 = MiniBatchKMeans(n_clusters=16, n_init=1000, random_state=12345, batch_size=100000).fit(dataGrid2)
km2_22 = MiniBatchKMeans(n_clusters=22, n_init=1000, random_state=12345, batch_size=100000).fit(dataGrid2)

# %%
spatialEnvSocialGrid2['clusters_all2_9'] = km2_9.labels_
spatialEnvSocialGrid2['clusters_all2_16'] = km2_16.labels_
spatialEnvSocialGrid2['clusters_all2_22'] = km2_22.labels_
spatialEnvSocialGrid2.to_parquet(folder + "spatialEnvSocialClusters2.pq")

# %%
ax = spatialEnvSocialGrid2.plot('clusters_all2_9', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction, Social Connectivity, and Spatial Signatures set 2 Clusters Grid (9)")
ax = spatialEnvSocialGrid2.plot('clusters_all2_16', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction, Social Connectivity, and Spatial Signatures set 2 Clusters Grid (16)")
ax = spatialEnvSocialGrid2.plot('clusters_all2_22', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction, Social Connectivity, and Spatial Signatures set 2 Clusters Grid (22)")

# %%
spatialEnvSocialGrid1.head(1)

# %%
clusternames1 = ["clusters_all1_11", "clusters_all1_15", "clusters_all1_18"]
clusternames2 = ["clusters_all2_9", "clusters_all2_16", "clusters_all2_22"]

# %%
tessEnvPredAreaAllClust1 = area_interpolate(spatialEnvSocialGrid1, tessEnvPredArea, intensive_variables=clusternames1, n_jobs=1)#, categorical_variables=clusternames1)
tessEnvPredAreaAllClust2 = area_interpolate(spatialEnvSocialGrid2, tessEnvPredArea, intensive_variables=clusternames2, n_jobs=1)#, categorical_variables=clusternames2)

# tessEnvPredArea[['tID', 'clusters',]].to_csv(folder + "FINAL_cluster_labels_v2.csv")

# %%
np.max(tessEnvPredAreaAllClust2["clusters_all2_9"])

# %%
np.min(tessEnvPredAreaAllClust2["clusters_all2_9"])

# %%
convert_dict = {'clusters_all1_11': int, 'clusters_all1_15': int,'clusters_all1_18': int}
tessEnvPredAreaAllClustData1 = np.round(tessEnvPredAreaAllClust1[['clusters_all1_11','clusters_all1_15','clusters_all1_18']])
tessEnvPredAreaAllClustData1 = tessEnvPredAreaAllClustData1.astype(convert_dict)
tessEnvPredAreaAllClust1 = gpd.GeoDataFrame(tessEnvPredAreaAllClustData1, geometry=tessEnvPredAreaAllClust1.geometry)

convert_dict = {'clusters_all2_9': int, 'clusters_all2_16': int,'clusters_all2_22': int}
tessEnvPredAreaAllClustData2 = np.round(tessEnvPredAreaAllClust2[['clusters_all2_9','clusters_all2_16','clusters_all2_22']])
tessEnvPredAreaAllClustData2 = tessEnvPredAreaAllClustData2.astype(convert_dict)
tessEnvPredAreaAllClust2 = gpd.GeoDataFrame(tessEnvPredAreaAllClustData2, geometry=tessEnvPredAreaAllClust2.geometry)


# %%
tessEnvPredAreaAllClust1['clusters_all1_11'].unique()

# %%
ax = tessEnvPredAreaAllClust1.plot('clusters_all1_11', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax = tessEnvPredAreaAllClust1.plot('clusters_all1_15', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax = tessEnvPredAreaAllClust1.plot('clusters_all1_18', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()

# %%
ax = tessEnvPredAreaAllClust2.plot('clusters_all2_9', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction, Social Connectivity, and Spatial Signatures set 2 Clusters ETCs (9)")

ax = spatialEnvSocialGrid2.plot('clusters_all2_9', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax.set_title("Environmnetal Prediction, Social Connectivity, and Spatial Signatures set 2 Clusters Grid (9)")

ax = tessEnvPredAreaAllClust2.plot('clusters_all2_16', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()
ax = tessEnvPredAreaAllClust2.plot('clusters_all2_22', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()

# %%
tessEnvPredAreaAllClust = tessEnvPredAreaAllClust1.merge(tessEnvPredAreaAllClust2, on="geometry")

# %%
tessEnvPredAreaAllClust.to_parquet(folder + "allClusters.pq")

# %%
tessEnvPredArea[['tID', 'clusters', 'clusters_form', 'clusters_function']].to_parquet(folder + "cluster_labels.pq")

# %% [markdown]
# #### Generate signatures
# 
# As a final step, we can generate a geometry representing spatial singatures as a combination of contiguous tessellation cells belonging to a single cluster. Since we have to dissolve a large number of polygons, we use dask_geopandas to parallelize the procedure.
# 
# First we create a local cluster.

# %% [markdown]
# We read only 'tessellation' columns and cluster labels. The resulting geometry will not be enclosed tessellation anymore so we can already rename it to generic "geometry".

# %%
# tess = gpd.read_parquet(folder + "tess.pq", columns=['tessellation']).rename_geometry("geometry")
clusters = pd.read_csv(folder + "FINAL_cluster_labels_v2.csv", index_col=0)

# %%
tessEnvPredArea['cluster'] = clusters.clusters.values

# %% [markdown]
# `dask_dissolve` mimics the behaviour of `geopandas.dissolve`, just based on parallel implementation using `dask-geopandas`.

# %%
# %%time
# ddf = dask_geopandas.from_geopandas(tessEnvPredArea.sort_values('cluster'), npartitions=64)
final = tessEnvPredArea.dissolve('cluster')

# %%
final

# %%
ax = final.plot('clusters', categorical=True, legend=True, figsize=(20, 20), cmap='tab20')
ax.set_axis_off()

# %% [markdown]
# Finally, we explode the geometries to get multiple singatures of the same kind instead of a single multipolygon. (The intial `reset_index` is here to avoid bug in geopandas.)

# %%
final.reset_index(drop=True).explode().reset_index(drop=True).to_parquet(folder + "signatures.pq")

# %% [markdown]
# Before we end, we close dask client.

# %% [markdown]
# # 4. Autocorrelation analysis

# %% [markdown]
# ## 4.1 Multi-variate local autocorrelation analysis

# %% [markdown]
# ### Spatial

# %%
used_characteristics_dict_v2 = {
    'sscCCo': 'circular compactness of ETC', #
    'sicCAR': 'covered area ratio of ETC', # building cover
    'lddNDe': 'local node density of street network',
    'sdsSPO': 'openness of street profile', #
    'ltbIBD': 'mean inter-building distance',
    'mtbAli': 'alignment of neighbouring buildings',
    'lieWCe': 'area-weighted ETCs of enclosure', # 
}

# %%
tess = gpd.read_parquet(folder + "tess.pq")
fullChunks = gpd.read_parquet(folder + "combTessChunks.pq")

# %%
tessWithClusters = gpd.read_parquet(folder + "finalSpatialClusteringEnvPredAll.pq")

# %%
# tessWithClusters.set_index(envPredTess.index, inplace=True)

# %%
pd.set_option('display.max_columns', None)
tessWithClusters.head(1)

# %%
envPredTess = gpd.read_parquet(folder + "envPredTess.pq")
# tessWithClusters = envPredTess.merge(tessWithClusters, on="geometry")
# envPredTess.head(1)

# %%
fullChunks.loc[fullChunks.ltbIBD==np.nan]

# %%
tessWithClusters.ltbIBD_q2

# %% [markdown]
# Next section is to make sure that multivariate autocorrelation can be run for all of the desired parameters however it is unneccessary for bivariate

# %%
fullChunks["sicCAR"].replace(np.nan, 0, inplace=True)
fullChunksCorr1 = fullChunks.copy()

# %%
fullChunks["sicCAR"].replace(np.nan, 0, inplace=True)


interbuilding = fullChunks.ltbIBD
alignment = fullChunks.mtbAli 
fullChunks["ltbIBD"].loc[interbuilding.isnull()] = tessWithClusters["ltbIBD_q2"].loc[interbuilding.isnull()]
fullChunks["mtbAli"].loc[alignment.isnull()] = tessWithClusters["mtbAli_q2"].loc[alignment.isnull()]
fullChunks["ltbIBD"].replace(np.nan, 0, inplace=True)
fullChunks["mtbAli"].replace(np.nan, 0, inplace=True)

fullChunksCorr2 = fullChunks.copy()

# %%
fullChunksCorr1 = fullChunksCorr1.merge(tessWithClusters[["geometry", "accessScore", "POP20", 
                                                        "QH", "QE", "QS", "Tsurf", "Evap",
                                                        "RO", "TotCh", "Drainage", "SMD", 
                                                        "heat_score", "water_score"]], 
                                                        on="geometry")

# %%
fullChunksCorr2 = fullChunksCorr2.merge(tessWithClusters[["geometry", "accessScore", "POP20", 
                                                        "QH", "QE", "QS", "Tsurf", "Evap",
                                                        "RO", "TotCh", "Drainage", "SMD", 
                                                        "heat_score", "water_score"]], 
                                                        on="geometry")

# %%
fullChunksCorr1.set_index(fullChunks.index, inplace=True)
fullChunksCorr2.set_index(fullChunks.index, inplace=True)
# fullChunksCorr.head(1)

# %%
wFullChunks1 = libpysal.weights.contiguity.Queen.from_dataframe(fullChunksCorr1)
wFullChunks2 = libpysal.weights.contiguity.Queen.from_dataframe(fullChunksCorr2)

# %%
spatial_chars = [
    'sdbAre', # 'area of building',
    'sdbPer', # 'perimeter of building',
    'sdbCoA', # 'courtyard area of building',
    'ssbCCo', # 'circular compactness of building', 
    'ssbCor', # 'corners of building',
    'ssbSqu', # 'squareness of building',
    'ssbERI', # 'equivalent rectangular index of building',
    'ssbElo', # 'elongation of building',
    'ssbCCM', # 'centroid - corner mean distance of building',
    'ssbCCD', # 'centroid - corner distance deviation of building',
    'stbOri', # 'orientation of building',
    'sdcLAL', # 'longest axis length of ETC',
    'sdcAre', # 'area of ETC', # not in score
    'sscCCo', # 'circular compactness of ETC', 
    'sscERI', # 'equivalent rectangular index of ETC',
    'stcOri', # 'orientation of ETC', #
    'sicCAR', # 'covered area ratio of ETC', # building cover
    'stbCeA', # 'cell alignment of building',
    'mtbAli', # 'alignment of neighbouring buildings',
    'mtbNDi', # 'mean distance between neighbouring buildings',
    'mtcWNe', # 'perimeter-weighted neighbours of ETC', 
    'mdcAre', # 'area covered by neighbouring cells',
    'ltcWRE', # 'weighted reached enclosures of ETC',
    'ltbIBD', # 'mean inter-building distance',
    'sdsSPW', # 'width of street profile',
    'sdsSWD', # 'width deviation of street profile',
    'sdsSPO', # 'openness of street profile', #
    'sdsLen', # 'length of street segment',
    'sssLin', # 'linearity of street segment', #
    'ldsMSL', # 'mean segment length within 3 steps',
    'mtdDeg', # 'node degree of junction',
    'lcdMes', # 'local meshedness of street network',
    'linP3W', # 'local proportion of 3-way intersections of street network',
    'linP4W', # 'local proportion of 4-way intersections of street network',
    'linPDE', # 'local proportion of cul-de-sacs of street network',
    'lcnClo', # 'local closeness of street network',
    'ldsCDL', # 'local cul-de-sac length of street network',
    'xcnSCl', # 'square clustering of street network',
    'mtdMDi', # 'mean distance to neighbouring nodes of street network',
    # 'clsCnt', # 'closeness centrality of nodes in street network', #dropped for ss
    # 'btwCnt', # 'betweenness centrality of nodes in street network',
    # 'strCnt', # 'straightness centrality of nodes in street network',
    'lddNDe', # 'local node density of street network',
    'linWID', # 'local degree weighted node density of street network',
    'stbSAl', # 'street alignment of building',
    'stcSAl', # 'street alignment of ETC', 
    'mtbSWR', # 'shared walls ratio of buildings',
    'sddAre', # 'area covered by node-attached ETCs',
    'sdsAre', # 'area covered by edge-attached ETCs',
    'sisBpM', # 'buildings per meter of street segment',
    'misCel', # 'reached ETCs by neighbouring segments',
    'mdsAre', # 'reached area by neighbouring segments',
    'lisCel', # 'reached ETCs by local street network',
    'ldsAre', # 'reached area by local street network',
    # 'bdShDv': 'shannon diversity index for buildings',
    'ltcRea', # 'reached ETCs by tessellation contiguity',
    'ltcAre', # 'reached area by tessellation contiguity',
    'ldeAre', # 'area of enclosure',
    'ldePer', # 'perimeter of enclosure',
    'lseCCo', # 'circular compactness of enclosure',
    'lseERI', # 'equivalent rectangular index of enclosure',
    'lseCWA', # 'compactness-weighted axis of enclosure',
    'lteOri', # 'orientation of enclosure',
    'lteWNB', # 'perimeter-weighted neighbours of enclosure',
    'lieWCe', # 'area-weighted ETCs of enclosure', # 
    'ldbPWL', # 'perimeter wall length of adjacent buildings',
    'libNCo', # 'number of courtyards within adjacent buildings',
    'misRea', # 'cells reached within neighbouring street segments'
]

# %%
for char in spatial_chars:
    if char in fullChunksCorr1.columns:
        spatial_lag_name = "w_" + char
        fullChunksCorr1[spatial_lag_name] = libpysal.weights.lag_spatial(wFullChunks1, fullChunksCorr1[char])
    if char in fullChunksCorr2.columns:
        spatial_lag_name = "w_" + char
        fullChunksCorr2[spatial_lag_name] = libpysal.weights.lag_spatial(wFullChunks2, fullChunksCorr2[char])
    else:
        continue

# %%
used_characteristics_v2

# %%
pd.set_option('display.max_columns', None)
fullChunksCorr1.head(1)

# %%
bv_Moran_FC1_soc_list = []
bv_Moran_FC2_soc_list = []

bv_Moran_FC1_heat_list = []
bv_Moran_FC2_heat_list = []

bv_Moran_FC1_water_list = []
bv_Moran_FC2_water_list = []

usedChars = list(used_characteristics_v2)

for char in usedChars:
    bv_Soc1 = Moran_Local_BV(fullChunksCorr1[char], fullChunksCorr1["accessScore"],wFullChunks1)    
    bv_Soc2 = Moran_Local_BV(fullChunksCorr2[char], fullChunksCorr2["accessScore"],wFullChunks2)    
    bv_Moran_FC1_soc_list.append(bv_Soc1)
    bv_Moran_FC2_soc_list.append(bv_Soc2)

    bv_Heat1 = Moran_Local_BV(fullChunksCorr1[char], fullChunksCorr1["heat_score"],wFullChunks1)
    bv_Heat2 = Moran_Local_BV(fullChunksCorr2[char], fullChunksCorr2["heat_score"],wFullChunks2)
    bv_Moran_FC1_heat_list.append(bv_Heat1)
    bv_Moran_FC2_heat_list.append(bv_Heat2)

    bv_Water1 = Moran_Local_BV(fullChunksCorr1[char], fullChunksCorr1["water_score"],wFullChunks1)
    bv_Water2 = Moran_Local_BV(fullChunksCorr2[char], fullChunksCorr2["water_score"],wFullChunks2)
    bv_Moran_FC1_water_list.append(bv_Water1)
    bv_Moran_FC2_water_list.append(bv_Water2)

# %%
bv_Moran_FC1_soc_dict = dict(zip(usedChars,bv_Moran_FC1_soc_list))
bv_Moran_FC2_soc_dict = dict(zip(usedChars,bv_Moran_FC2_soc_list))
bv_Moran_FC1_heat_dict = dict(zip(usedChars,bv_Moran_FC1_heat_list))
bv_Moran_FC2_heat_dict = dict(zip(usedChars,bv_Moran_FC2_heat_list))
bv_Moran_FC1_water_dict = dict(zip(usedChars,bv_Moran_FC1_water_list))
bv_Moran_FC2_water_dict = dict(zip(usedChars,bv_Moran_FC2_water_list))


# %%
bv_Moran_FC1_soc_dict[usedChars[0]].p_sim

# %%
scores = ["accessScore", "heat_score", "water_score"]

# %%
bv1 = Moran_Local_BV.by_col(df=fullChunksCorr1, x=usedChars, y=scores, w=wFullChunks1)    
bv2 = Moran_Local_BV.by_col(df=fullChunksCorr2, x=usedChars, y=scores, w=wFullChunks2)    


# %%
"access_score_"+ usedChars[0] + "_quadrant"


# %%
[usedChars[2]] + scores

# %%
bv2IvalCheck = bv2[usedChars + scores].corr(method="kendall")


# %%
bv2sdsSPO = bv2[[usedChars[0]] + scores].corr(method="kendall")
bv2sicCAR = bv2[[usedChars[1]] + scores].corr(method="kendall")
bv2sscCCo = bv2[[usedChars[2]] + scores].corr(method="kendall")
bv2lieWCe = bv2[[usedChars[3]] + scores].corr(method="kendall")
bv2mtbAli = bv2[[usedChars[4]] + scores].corr(method="kendall")
bv2lddNDe = bv2[[usedChars[5]] + scores].corr(method="kendall")
bv2ltbIBD = bv2[[usedChars[6]] + scores].corr(method="kendall")

# %%
bv2ltbIBD

# %%
def convertQuadrants (df, quadrant, p_sim, p_val=0.05):
     # Convert all non-significant quadrants to zero
    df[quadrant] = np.where(df[p_sim] > p_val, 0, df[quadrant])

    # Get more informative descriptions
    df[quadrant] = df[quadrant].replace(
    to_replace = {
        0: "Not significant",
        1: "High-high",
        2: "Low-high",
        3: "Low-low",
        4: "High-low"
    }
    )
    return df

# %%
fullChunksBVCorr1 = fullChunksCorr1.copy()
fullChunksBVCorr2 = fullChunksCorr2.copy()

quadList = []
psimList = []

for char in usedChars:
    socTitleQ = "access_score_"+ char + "_quadrant"
    socTitlePsim = "access_score_"+ char + "_p_sim"
    heatTitleQ = "heat_score_"+ char + "_quadrant"
    heatTitlePsim = "heat_score_"+ char + "_p_sim"
    waterTitleQ = "water_score_"+ char + "_quadrant"
    waterTitlePsim = "water_score_"+ char + "_p_sim"
    quadList.append(socTitleQ)
    quadList.append(heatTitleQ)
    quadList.append(waterTitleQ)
    psimList.append(socTitlePsim)
    psimList.append(heatTitlePsim)
    psimList.append(waterTitlePsim)

    fullChunksBVCorr1[socTitleQ] = bv_Moran_FC1_soc_dict[char].q
    fullChunksBVCorr2[socTitleQ] = bv_Moran_FC2_soc_dict[char].q
    fullChunksBVCorr1[socTitlePsim] = bv_Moran_FC1_soc_dict[char].p_sim
    fullChunksBVCorr2[socTitlePsim] = bv_Moran_FC2_soc_dict[char].p_sim
    fullChunksBVCorr1 = convertQuadrants(fullChunksBVCorr1, socTitleQ, socTitlePsim)
    fullChunksBVCorr2 = convertQuadrants(fullChunksBVCorr2, socTitleQ, socTitlePsim)

    fullChunksBVCorr1[heatTitleQ] = bv_Moran_FC1_heat_dict[char].q
    fullChunksBVCorr2[heatTitleQ] = bv_Moran_FC2_heat_dict[char].q
    fullChunksBVCorr1[heatTitlePsim] = bv_Moran_FC1_heat_dict[char].p_sim
    fullChunksBVCorr2[heatTitlePsim] = bv_Moran_FC2_heat_dict[char].p_sim
    fullChunksBVCorr1 = convertQuadrants(fullChunksBVCorr1, heatTitleQ, heatTitlePsim)
    fullChunksBVCorr2 = convertQuadrants(fullChunksBVCorr2, heatTitleQ, heatTitlePsim)

    fullChunksBVCorr1[waterTitleQ] = bv_Moran_FC1_water_dict[char].q
    fullChunksBVCorr2[waterTitleQ] = bv_Moran_FC2_water_dict[char].q
    fullChunksBVCorr1[waterTitlePsim] = bv_Moran_FC1_water_dict[char].p_sim
    fullChunksBVCorr2[waterTitlePsim] = bv_Moran_FC2_water_dict[char].p_sim
    fullChunksBVCorr1 = convertQuadrants(fullChunksBVCorr1, waterTitleQ, waterTitlePsim)
    fullChunksBVCorr2 = convertQuadrants(fullChunksBVCorr2, waterTitleQ, waterTitlePsim)


# %%
fullChunksBVCorr2_v2 = fullChunksCorr2.copy()

quadList = []
psimList = []

for char in usedChars:
    socTitleQ = "access_score_"+ char + "_quadrant"
    socTitlePsim = "access_score_"+ char + "_p_sim"
    heatTitleQ = "heat_score_"+ char + "_quadrant"
    heatTitlePsim = "heat_score_"+ char + "_p_sim"
    waterTitleQ = "water_score_"+ char + "_quadrant"
    waterTitlePsim = "water_score_"+ char + "_p_sim"
    quadList.append(socTitleQ)
    quadList.append(heatTitleQ)
    quadList.append(waterTitleQ)
    psimList.append(socTitlePsim)
    psimList.append(heatTitlePsim)
    psimList.append(waterTitlePsim)

    fullChunksBVCorr2_v2[socTitleQ] = bv_Moran_FC2_soc_dict[char].q
    fullChunksBVCorr2_v2[socTitlePsim] = bv_Moran_FC2_soc_dict[char].p_sim
    fullChunksBVCorr2_v2 = convertQuadrants(fullChunksBVCorr2_v2, socTitleQ, socTitlePsim, p_val=0.01)

    fullChunksBVCorr2_v2[heatTitleQ] = bv_Moran_FC2_heat_dict[char].q
    fullChunksBVCorr2_v2[heatTitlePsim] = bv_Moran_FC2_heat_dict[char].p_sim
    fullChunksBVCorr2_v2 = convertQuadrants(fullChunksBVCorr2_v2, heatTitleQ, heatTitlePsim, p_val=0.01)

    fullChunksBVCorr2_v2[waterTitleQ] = bv_Moran_FC2_water_dict[char].q
    fullChunksBVCorr2_v2[waterTitlePsim] = bv_Moran_FC2_water_dict[char].p_sim
    fullChunksBVCorr2_v2 = convertQuadrants(fullChunksBVCorr2_v2, waterTitleQ, waterTitlePsim, p_val=0.01)


# %%
savedVals = usedChars + ["geometry","uID", "eID", "clsCnt","btwCnt","strCnt", "accessScore", "POP20", 
                        "QH", "QE", "QS", "Tsurf", "Evap", "RO", "TotCh", "Drainage", "SMD", 
                        "heat_score", "water_score", "w_sdsSPO","w_sicCAR", "w_sscCCo", 
                        "w_lieWCe", "w_mtbAli", "w_lddNDe", "w_ltbIBD"] + quadList + psimList


# %%
fullChunksBVCorr1Save = fullChunksBVCorr1[savedVals]
fullChunksBVCorr2Save = fullChunksBVCorr2[savedVals]

fullChunksBVCorr1Save.to_parquet(folder + "BV_autocorrelation_scores_spatial_v1.pq")
fullChunksBVCorr2Save.to_parquet(folder + "BV_autocorrelation_scores_spatial_v2.pq")

# %%
fullChunksBVCorrSave = fullChunksBVCorr2_v2[savedVals]

fullChunksBVCorrSave.to_parquet(folder + "BV_autocorrelation_scores_spatial_v2_p_0_01.pq")

# %%
wFullChunksCorrection = libpysal.weights.contiguity.Queen.from_dataframe(fullChunksBVCorr2Save)

bv_Moran_FC_soc_list = []

usedChars = list(used_characteristics_v2)

for char in usedChars:
    bv_Soc = Moran_Local_BV(fullChunksBVCorr2Save[char], fullChunksBVCorr2Save["accessScore"],wFullChunksCorrection)    
    bv_Moran_FC_soc_list.append(bv_Soc)

bv_Moran_FC_soc_dict = dict(zip(usedChars,bv_Moran_FC_soc_list))

# %%
fullChunksBVCorr = fullChunksBVCorr2Save.copy()

quadList = []
psimList = []

for char in usedChars:
    socTitleQ = "access_score_"+ char + "_quadrant"
    socTitlePsim = "access_score_"+ char + "_p_sim"
    # heatTitleQ = "heat_score_"+ char + "_quadrant"
    # heatTitlePsim = "heat_score_"+ char + "_p_sim"
    # waterTitleQ = "water_score_"+ char + "_quadrant"
    # waterTitlePsim = "water_score_"+ char + "_p_sim"
    quadList.append(socTitleQ)
    # quadList.append(heatTitleQ)
    # quadList.append(waterTitleQ)
    psimList.append(socTitlePsim)
    # psimList.append(heatTitlePsim)
    # psimList.append(waterTitlePsim)

    fullChunksBVCorr[socTitleQ] = bv_Moran_FC_soc_dict[char].q
    fullChunksBVCorr[socTitlePsim] = bv_Moran_FC_soc_dict[char].p_sim
    fullChunksBVCorr = convertQuadrants(fullChunksBVCorr, socTitleQ, socTitlePsim)

    # fullChunksBVCorr1[heatTitleQ] = bv_Moran_FC1_heat_dict[char].q
    # fullChunksBVCorr1[heatTitlePsim] = bv_Moran_FC1_heat_dict[char].p_sim
    # fullChunksBVCorr1 = convertQuadrants(fullChunksBVCorr1, heatTitleQ, heatTitlePsim)

    # fullChunksBVCorr1[waterTitleQ] = bv_Moran_FC1_water_dict[char].q
    # fullChunksBVCorr1[waterTitlePsim] = bv_Moran_FC1_water_dict[char].p_sim
    # fullChunksBVCorr1 = convertQuadrants(fullChunksBVCorr1, waterTitleQ, waterTitlePsim)


# %%
fullChunksBVCorr.to_parquet(folder + "BV_autocorrelation_scores_good_accessScore.pq")

# %%
from splot.esda import plot_local_autocorrelation, plot_moran_bv

# %% [markdown]
# ### Plotting BV autocorrelation for different features (set 1)

# %% [markdown]
# #### Social Sustainability

# %%
fg, axs = plot_local_autocorrelation(bv_Moran_FC1_soc_list[0], fullChunksCorr1, attribute=usedChars[0], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[0]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()

# %%
fg, axs = plot_local_autocorrelation(bv_Moran_FC1_soc_list[1], fullChunksCorr1, attribute=usedChars[1], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[1]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_soc_list[2], fullChunksCorr1, attribute=usedChars[2], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[2]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_soc_list[3], fullChunksCorr1, attribute=usedChars[3], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[3]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_soc_list[4], fullChunksCorr1, attribute=usedChars[4], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[4]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_soc_list[5], fullChunksCorr1, attribute=usedChars[5], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[5]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_soc_list[6], fullChunksCorr1, attribute=usedChars[6], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[6]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()

# %% [markdown]
# #### Heat Score

# %%
fg, axs = plot_local_autocorrelation(bv_Moran_FC1_heat_list[0], fullChunksCorr1, attribute=usedChars[0], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[0]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()

# %%
fg, axs = plot_local_autocorrelation(bv_Moran_FC1_heat_list[1], fullChunksCorr1, attribute=usedChars[1], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[1]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_heat_list[2], fullChunksCorr1, attribute=usedChars[2], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[2]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_heat_list[3], fullChunksCorr1, attribute=usedChars[3], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[3]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_heat_list[4], fullChunksCorr1, attribute=usedChars[4], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[4]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_heat_list[5], fullChunksCorr1, attribute=usedChars[5], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[5]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_heat_list[6], fullChunksCorr1, attribute=usedChars[6], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[6]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()

# %% [markdown]
# #### Water Score

# %%
fg, axs = plot_local_autocorrelation(bv_Moran_FC1_water_list[0], fullChunksCorr1, attribute=usedChars[0], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[0]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()

# %%
fg, axs = plot_local_autocorrelation(bv_Moran_FC1_water_list[1], fullChunksCorr1, attribute=usedChars[1], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[1]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_water_list[2], fullChunksCorr1, attribute=usedChars[2], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[2]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_water_list[3], fullChunksCorr1, attribute=usedChars[3], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[3]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_water_list[4], fullChunksCorr1, attribute=usedChars[4], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[4]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_water_list[5], fullChunksCorr1, attribute=usedChars[5], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[5]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_water_list[6], fullChunksCorr1, attribute=usedChars[6], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[6]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()

# %% [markdown]
# ### Plotting BV autocorrelation for different features (set 2)

# %%
fg, axs = plot_local_autocorrelation(bv_Moran_FC1_soc_list[0], fullChunksCorr1, attribute=usedChars[0], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[0]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()

# %%
fg, axs = plot_local_autocorrelation(bv_Moran_FC1_soc_list[1], fullChunksCorr1, attribute=usedChars[1], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[1]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_soc_list[2], fullChunksCorr1, attribute=usedChars[2], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[2]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_soc_list[3], fullChunksCorr1, attribute=usedChars[3], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[3]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_soc_list[4], fullChunksCorr1, attribute=usedChars[4], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[4]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_soc_list[5], fullChunksCorr1, attribute=usedChars[5], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[5]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_soc_list[6], fullChunksCorr1, attribute=usedChars[6], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[6]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()

# %%
fg, axs = plot_local_autocorrelation(bv_Moran_FC1_heat_list[0], fullChunksCorr1, attribute=usedChars[0], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[0]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()

# %%
fg, axs = plot_local_autocorrelation(bv_Moran_FC1_heat_list[1], fullChunksCorr1, attribute=usedChars[1], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[1]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_heat_list[2], fullChunksCorr1, attribute=usedChars[2], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[2]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_heat_list[3], fullChunksCorr1, attribute=usedChars[3], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[3]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_heat_list[4], fullChunksCorr1, attribute=usedChars[4], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[4]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_heat_list[5], fullChunksCorr1, attribute=usedChars[5], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[5]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_heat_list[6], fullChunksCorr1, attribute=usedChars[6], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[6]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()

# %%
fg, axs = plot_local_autocorrelation(bv_Moran_FC1_water_list[0], fullChunksCorr1, attribute=usedChars[0], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[0]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()

# %%
fg, axs = plot_local_autocorrelation(bv_Moran_FC1_water_list[1], fullChunksCorr1, attribute=usedChars[1], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[1]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_water_list[2], fullChunksCorr1, attribute=usedChars[2], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[2]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_water_list[3], fullChunksCorr1, attribute=usedChars[3], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[3]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_water_list[4], fullChunksCorr1, attribute=usedChars[4], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[4]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_water_list[5], fullChunksCorr1, attribute=usedChars[5], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[5]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC1_water_list[6], fullChunksCorr1, attribute=usedChars[6], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[6]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()

# %% [markdown]
# #### Social Sustainability

# %%
usedChars[0]

# %%
fg, axs = plot_local_autocorrelation(bv_Moran_FC2_soc_list[0], fullChunksCorr2, attribute=usedChars[0], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[0]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()

# %%
usedChars[1]

# %%
fg, axs = plot_local_autocorrelation(bv_Moran_FC2_soc_list[1], fullChunksCorr2, attribute=usedChars[1], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[1]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()


# %%
usedChars[2]

# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC2_soc_list[2], fullChunksCorr2, attribute=usedChars[2], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[2]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()


# %%
usedChars[3]

# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC2_soc_list[3], fullChunksCorr2, attribute=usedChars[3], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[3]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC2_soc_list[4], fullChunksCorr2, attribute=usedChars[4], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[4]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC2_soc_list[5], fullChunksCorr2, attribute=usedChars[5], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[5]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC2_soc_list[6], fullChunksCorr2, attribute=usedChars[6], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[6]] + ' with access score'
fg.suptitle(titleTxt)
plt.show()

# %% [markdown]
# #### Heat Score

# %%
fg, axs = plot_local_autocorrelation(bv_Moran_FC2_heat_list[0], fullChunksCorr2, attribute=usedChars[0], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[0]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()

# %%
fg, axs = plot_local_autocorrelation(bv_Moran_FC2_heat_list[1], fullChunksCorr2, attribute=usedChars[1], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[1]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC2_heat_list[2], fullChunksCorr2, attribute=usedChars[2], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[2]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC2_heat_list[3], fullChunksCorr2, attribute=usedChars[3], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[3]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC2_heat_list[4], fullChunksCorr2, attribute=usedChars[4], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[4]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC2_heat_list[5], fullChunksCorr2, attribute=usedChars[5], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[5]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC2_heat_list[6], fullChunksCorr2, attribute=usedChars[6], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[6]] + ' with heat score'
fg.suptitle(titleTxt)
plt.show()

# %% [markdown]
# #### Water Score

# %%
fg, axs = plot_local_autocorrelation(bv_Moran_FC2_water_list[0], fullChunksCorr2, attribute=usedChars[0], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[0]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()

# %%
fg, axs = plot_local_autocorrelation(bv_Moran_FC2_water_list[1], fullChunksCorr2, attribute=usedChars[1], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[1]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC2_water_list[2], fullChunksCorr2, attribute=usedChars[2], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[2]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC2_water_list[3], fullChunksCorr2, attribute=usedChars[3], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[3]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC2_water_list[4], fullChunksCorr2, attribute=usedChars[4], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[4]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC2_water_list[5], fullChunksCorr2, attribute=usedChars[5], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[5]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()


# %%

fg, axs = plot_local_autocorrelation(bv_Moran_FC2_water_list[6], fullChunksCorr2, attribute=usedChars[6], p=0.05, legend=True)#, mask=['Ain'], quadrant=1)
titleTxt = used_characteristics_dict_v2[usedChars[6]] + ' with water score'
fg.suptitle(titleTxt)
plt.show()

# %% [markdown]
# ### Centrality Autocorrelation

# %%
fullChunksBVCorr1Save = gpd.read_parquet(folder + "BV_autocorrelation_scores_spatial_v1.pq")
fullChunksBVCorr2Save = gpd.read_parquet(folder + "BV_autocorrelation_scores_spatial_v2.pq")

# %%
centralityAutoCorr = fullChunksBVCorr1Save[["clsCnt","btwCnt","strCnt","accessScore","POP20","geometry"]]

# %%
maxPopScale = np.median(centralityAutoCorr['POP20'])#np.percentile(popHousing_gdf['POP20'],75)
DenScale = np.mean(centralityAutoCorr['POP20']) + np.std(centralityAutoCorr['POP20'])
minPopScale = 1- np.percentile(centralityAutoCorr['POP20'],25)/DenScale


# %% [markdown]
# Correction to the access score computation

# %%
pop = centralityAutoCorr['POP20']
pop = minPopScale + pop/DenScale
maxScale = maxPopScale*np.ones(len(pop))
pop = np.minimum(pop,maxScale)

# %%
clsNorm = (centralityAutoCorr["clsCnt"] - np.min(centralityAutoCorr["clsCnt"]))/(np.max(centralityAutoCorr["clsCnt"]) - np.min(centralityAutoCorr["clsCnt"]))
btwNorm = (centralityAutoCorr["btwCnt"] - np.min(centralityAutoCorr["btwCnt"]))/(np.max(centralityAutoCorr["btwCnt"]) - np.min(centralityAutoCorr["btwCnt"]))
strNorm = (centralityAutoCorr["strCnt"] - np.min(centralityAutoCorr["strCnt"]))/(np.max(centralityAutoCorr["strCnt"]) - np.min(centralityAutoCorr["strCnt"]))

# %%
accessScoreNew = (clsNorm + btwNorm + strNorm)*pop

# %%
centralityAutoCorr["accessScoreNew"] = accessScoreNew

# %%
wCentrality = libpysal.weights.contiguity.Queen.from_dataframe(centralityAutoCorr)


# %%
centralityAutoCorrList = ["clsCnt","btwCnt","strCnt"]
clsCentrMoran = Moran_Local(centralityAutoCorr["clsCnt"], w=wCentrality)    
btwCentrMoran = Moran_Local(centralityAutoCorr["btwCnt"], w=wCentrality)    
strCentrMoran = Moran_Local(centralityAutoCorr["strCnt"], w=wCentrality)    


# %%
interconnectivityMoran = Moran_Local(centralityAutoCorr["accessScore"], w=wCentrality)    


# %%
interconnectivity2Moran = Moran_Local(centralityAutoCorr["accessScoreNew"], w=wCentrality)    


# %%
centralityAutoCorr["interconnectivity_quadrant"] = interconnectivityMoran.q
centralityAutoCorr["interconnectivity_psim"] = interconnectivityMoran.p_sim
centralityAutoCorr = convertQuadrants(centralityAutoCorr, "interconnectivity_quadrant", "interconnectivity_psim")

# %%
centralityAutoCorr["interconnectivityNew_quadrant"] = interconnectivity2Moran.q
centralityAutoCorr["interconnectivityNew_psim"] = interconnectivity2Moran.p_sim
centralityAutoCorr = convertQuadrants(centralityAutoCorr, "interconnectivityNew_quadrant", "interconnectivityNew_psim")

# %%
centralityAutoCorr["clsCnt_quadrant"] = clsCentrMoran.q
centralityAutoCorr["btwCnt_quadrant"] = btwCentrMoran.q
centralityAutoCorr["strCnt_quadrant"] = strCentrMoran.q
centralityAutoCorr["clsCnt_psim"] = clsCentrMoran.p_sim
centralityAutoCorr["btwCnt_psim"] = btwCentrMoran.p_sim
centralityAutoCorr["strCnt_psim"] = strCentrMoran.p_sim
centralityAutoCorr = convertQuadrants(centralityAutoCorr, "clsCnt_quadrant", "clsCnt_psim")
centralityAutoCorr = convertQuadrants(centralityAutoCorr, "btwCnt_quadrant", "btwCnt_psim")
centralityAutoCorr = convertQuadrants(centralityAutoCorr, "strCnt_quadrant", "strCnt_psim")


# %% [markdown]
# ### Tree Crown Density

# %%
tessWTree = fullChunks.copy()

# %%
tessWTree.head(1)

# %%
treeCentroids = gpd.read_file(SOC_ANALYSIS_folder + "treeCentroidsFinal.gpkg")

# %% [markdown]
# Calculate number of trees on each tesselation cell

# %%
treeTess = tessWTree.join(
    gpd.sjoin(treeCentroids, tessWTree).groupby("index_right").size().rename("tree_score"),
    how="left",
)
treeTess["tree_score"] = treeTess["tree_score"].fillna(0.0)

# %%
test = treeTess.copy()
test["tree_score"].iloc[0] = treeTess["tree_score"].iloc[0] + 1.0

# %%
test.head(1)

# %%
test["geometry"].iloc[0].distance(treeCentroids.geometry.iloc[0])


# %%
def treeScore(gdf, treeC):
    tree_df = gdf.copy()
    it = iter(range(len(tree_df)))
    for i in it:
        it2 = iter(range(len(treeC)))
        for i2 in it2:
            dist = tree_df["geometry"].iloc[i].distance(treeC.geometry.iloc[i2])
            if dist < 20 and dist > 0:
                tree_df["tree_score"].iloc[i] = tree_df["tree_score"].iloc[i] + 1/dist
        # Start by counting number of points within each polygon
        # then find distance from all edges connected to the ETC to trees accessible within 5 steps
        # add that to tree count thereby providing you with tree crown density
    return tree_df

# %%
treeTess = treeScore(treeTess, treeCentroids)
treeTess.head(5)

# %%
np.max(treeTess["tree_score"])

# %%
wTrees = libpysal.weights.contiguity.Queen.from_dataframe(treeTess)
treeDensityMoran = Moran_Local(treeTess["tree_score"], w=wTrees) 

# %%
treeTess["treeDensity_quadrant"] = treeDensityMoran.q
treeTess["treeDensity_psim"] = treeDensityMoran.p_sim
treeTess = convertQuadrants(treeTess, "treeDensity_quadrant", "treeDensity_psim")

# %%
treeT = treeTess.merge(tessWithClusters[["geometry", "accessScore"]], on="geometry")

# %%
treeT[["bdShDv","accessScore"]].corr(method='kendall')

# %%
treeTess.to_parquet(folder + "treeDensity.pq")

# %%
centralityAutoCorr.to_parquet(folder + "centralityAutoCorrelationNew.pq")

# %%
edges_all = dd.read_parquet(folder + "edges/*")
edges_all = edges_all.compute()

# %%
edges_all.head(1)

# %% [markdown]
# ### MV Autocorrelation

# %% [markdown]
# What will be of interest is to do a simple correlation calculation between scoring and the different spatial features, before doing an autocorrelation to find the main clusters of those spatial features.
# 
# First finding the Z values per spatial character available. Then using the multivariate local spatial autocorrelation as defined in [Yamada, 2024] which amounts to 
# 
# ∑𝑖=1𝑛𝐼𝑖=tr(1𝑝𝑞𝑾𝒁𝒁⊤)=1𝑝𝑞tr(𝒁⊤𝑾𝒁)=𝐼.

# %%
moran_z_df = pd.DataFrame(data=fullChunksCorr.index)
moran_I = []
moran_EI = []
cols = []
for char in spatial_chars:
    if char in fullChunksCorr.columns:
        moran_ = Moran(fullChunksCorr[char],wFullChunks)
        moran_z_df[char] = moran_.z
        moran_I.append(moran_.I)
        moran_EI.append(moran_.EI)
        cols.append(char) 
    else:
        continue

# %%
moran_z_df.set_index("tID", inplace=True)
moran_z_df2 = moran_z_df.dropna(axis=1,how='all').copy()


# %%
moran_z_df.head(1)

# %%
usedChars = list(used_characteristics_v2)
Z_df = pd.DataFrame(data=fullChunksCorr.index)
Z_df.set_index("tID", inplace=True)


for char in spatial_chars:
    if (char in usedChars) and (char in moran_z_df2.columns):
        Z_df[char] = moran_z_df2[char]

# %%
q = wFullChunks.s0
p = float(len(Z_df.columns))
scale_pq = 1/(p*q)

# %%
wFullChunks.transform='b'

# %%
Z_mat = Z_df.to_numpy()
W_mat =  wFullChunks.full()[0]

# %%


# %% [markdown]
# According to Yamada 2024, this fourth column is there to indicate how all the parameters relate, however the hotspots are still local to each parameter

# %%
Z_mat = Z_df.to_numpy()
W_mat =  wFullChunks.full()[0]

I_local_MV = []

input = iter(range(len(Z_mat)))

for inp in input:
    Z_i = Z_mat[inp, :]
    W_i = W_mat[inp, :]
    I_i = scale_pq*np.matmul(np.matmul(W_i, Z_mat), Z_i.T)

    I_local_MV.append(I_i)


# %%
I_global_MV = scale_pq*np.trace(np.matmul(np.matmul(Z_mat.T, W_mat), Z_mat))

# %%
I_local_MV_Moran_df = pd.DataFrame(data=I_local_MV, index=fullChunksCorr.index)

# %%
np.min(I_local_MV)

# %%
used_characteristics_v2 # sscCCo sdsSPO lddNDe lieWCe

# %%
moran_df = pd.DataFrame(data=[moran_I,moran_EI])#, columns=cols)
# moran_df = moran_df.T
moran_df.columns=cols
moran_df
# moran_I_df.replace(np.nan, 0)

# %% [markdown]
# Start with correlations tomorrow, 

# %%
circCompactHeatCorr = spatialEnvSocialTess8wPop[["sscCCo_q2", "heat_score"]].corr(method="kendall")


# %%
# Good for final display but too costly
# tessWithClusters.explore(
#     tiles='cartodbpositron',
#     column='clusters_form_set5_social_conn_EP_all_11',
#     height='60%',
#     width='60%',
#     scheme='naturalbreaks',
#     cmap='Oranges',
#     style_kwds={
#         'stroke': True,
#         'edgecolor': 'k',
#         'linewidth': 0.03,
#         'fillOpacity': 1
#         },
#     legend_kwds={
#         'colorbar': False,
#         'fmt': '{:.0f}'
#         }
#     )

# %% [markdown]
# Need to get the environmental results for the tesselation to do a proper spatial autocorrelation analysis

# %% [markdown]
# # 5. Sensitivity Analysis

# %%
# original value for 100m 13
original = glob.glob(SA_folder + "100m_13_orig/*.txt")

# Model outputs from SA
anthro_minus = glob.glob(SA_folder + "100m_13_anthropogenic_minus/*.txt") ##
anthro_plus = glob.glob(SA_folder + "100m_13_anthropogenic_Plus/*.txt") ##
biogen_minus = glob.glob(SA_folder + "100m_13_Biogen_minus/*.txt") # Not impactful
biogen_plus = glob.glob(SA_folder + "100m_13_Biogen_Plus/*.txt") # Not impactful
conductance_minus = glob.glob(SA_folder + "100m_13_Conductance_minus/*.txt") ##
conductance_plus = glob.glob(SA_folder + "100m_13_Conductance_Plus/*.txt") ##
ESTM_minus = glob.glob(SA_folder + "100m_13_ESTM_minus/*.txt") # Not impactful
ESTM_plus = glob.glob(SA_folder + "100m_13_ESTM_Plus/*.txt") # Not impactful
irrigation_minus = glob.glob(SA_folder + "100m_13_Irrigation_minus/*.txt") # Not impactful
irrigation_plus = glob.glob(SA_folder + "100m_13_Irrigation_Plus/*.txt") # Not impactful
OHM_minus = glob.glob(SA_folder + "100m_13_OHM_minus/*.txt") ##
OHM_plus = glob.glob(SA_folder + "100m_13_OHM_Plus/*.txt") ##
Profiles_minus = glob.glob(SA_folder + "100m_13_Profiles_minus/*.txt") # Not impactful
Profiles_plus = glob.glob(SA_folder + "100m_13_Profiles_Plus/*.txt") # Not impactful
soil_char_minus = glob.glob(SA_folder + "100m_13_soil_char_minus/*.txt") ##
soil_char_plus = glob.glob(SA_folder + "100m_13_soil_char_Plus/*.txt") ##
surface_reflectance_minus = glob.glob(SA_folder + "100m_13_surface_reflectance_minus/*.txt") ##
surface_reflectance_plus = glob.glob(SA_folder + "100m_13_surface_reflectance_Plus/*.txt") ##
surface_water_minus = glob.glob(SA_folder + "100m_13_surface_water_minus/*.txt") ##
surface_water_plus = glob.glob(SA_folder + "100m_13_surface_water_Plus/*.txt") ##
Veg_minus = glob.glob(SA_folder + "100m_13_Veg_minus/*.txt")
Veg_plus = glob.glob(SA_folder + "100m_13_Veg_Plus/*.txt")

# %%
SA_inputs = glob.glob(SA_IN_folder + "*.txt")

# %%
SA_inputs_list = []

SA_inputs_iter = iter(range(len(SA_inputs)))

for i in SA_inputs_iter:
    SA_inputs_list.append(pd.read_table(SA_inputs[i]).rename(columns=lambda x: x.strip()))


# %%
anthro_inputs = SA_inputs_list[0]
anthro_inputs.set_index('Code',inplace=True)
conductance_inputs = SA_inputs_list[1]
conductance_inputs.set_index('Code',inplace=True)
OHM_inputs = SA_inputs_list[2]
OHM_inputs.set_index('Code',inplace=True)
soil_char_inputs = SA_inputs_list[3]
soil_char_inputs.set_index('Code',inplace=True)
surface_reflectance_inputs = SA_inputs_list[4]
surface_reflectance_inputs.set_index('Code',inplace=True)
surface_water_inputs = SA_inputs_list[5]
surface_water_inputs.set_index('Code',inplace=True)
Veg_inputs = SA_inputs_list[6]
Veg_inputs.set_index('Code',inplace=True)


# %%
original_list = []

saInputs = iter(range(len(original)))

for inp in saInputs:
    original_list.append(pd.read_table(original[inp]).rename(columns=lambda x: x.strip()))


# %%
anthro_list = []
biogen_list = []
conductance_list = []
ESTM_list = []
irrigation_list = []
OHM_list = []
Profiles_list = []
soil_char_list = []
surface_reflectance_list = []
surface_water_list = []
Veg_list = []

anthro_minus_list = []
biogen_minus_list = []
conductance_minus_list = []
ESTM_minus_list = []
irrigation_minus_list = []
OHM_minus_list = []
Profiles_minus_list = []
soil_char_minus_list = []
surface_reflectance_minus_list = []
surface_water_minus_list = []
Veg_minus_list = []

anthro_plus_list = []
biogen_plus_list = []
conductance_plus_list = []
ESTM_plus_list = []
irrigation_plus_list = []
OHM_plus_list = []
Profiles_plus_list = []
soil_char_plus_list = []
surface_reflectance_plus_list = []
surface_water_plus_list = []
Veg_plus_list = []

saInputs = iter(range(len(original)))

for inp in saInputs:
    anthro_list.append([pd.read_table(anthro_minus[inp]).rename(columns=lambda x: x.strip()),
                        pd.read_table(anthro_plus[inp]).rename(columns=lambda x: x.strip())])
    biogen_list.append([pd.read_table(biogen_minus[inp]).rename(columns=lambda x: x.strip()),
                        pd.read_table(biogen_plus[inp]).rename(columns=lambda x: x.strip())])
    conductance_list.append([pd.read_table(conductance_minus[inp]).rename(columns=lambda x: x.strip()),
                              pd.read_table(conductance_plus[inp]).rename(columns=lambda x: x.strip())])
    ESTM_list.append([pd.read_table(ESTM_minus[inp]).rename(columns=lambda x: x.strip()), 
                      pd.read_table(ESTM_plus[inp]).rename(columns=lambda x: x.strip())])
    irrigation_list.append([pd.read_table(irrigation_minus[inp]).rename(columns=lambda x: x.strip()),
                            pd.read_table(irrigation_plus[inp]).rename(columns=lambda x: x.strip())])
    OHM_list.append([pd.read_table(OHM_minus[inp]).rename(columns=lambda x: x.strip()),
                     pd.read_table(OHM_plus[inp]).rename(columns=lambda x: x.strip())])
    Profiles_list.append([pd.read_table(Profiles_minus[inp]).rename(columns=lambda x: x.strip()),
                          pd.read_table(Profiles_plus[inp]).rename(columns=lambda x: x.strip())])
    soil_char_list.append([pd.read_table(soil_char_minus[inp]).rename(columns=lambda x: x.strip()),
                           pd.read_table(soil_char_plus[inp]).rename(columns=lambda x: x.strip())])
    surface_reflectance_list.append([pd.read_table(surface_reflectance_minus[inp]).rename(columns=lambda x: x.strip()),
                                     pd.read_table(surface_reflectance_plus[inp]).rename(columns=lambda x: x.strip())])
    surface_water_list.append([pd.read_table(surface_water_minus[inp]).rename(columns=lambda x: x.strip()),
                               pd.read_table(surface_water_plus[inp]).rename(columns=lambda x: x.strip())])
    Veg_list.append([pd.read_table(Veg_minus[inp]).rename(columns=lambda x: x.strip()),
                     pd.read_table(Veg_plus[inp]).rename(columns=lambda x: x.strip())])
    
    anthro_minus_list.append(pd.read_table(anthro_minus[inp]).rename(columns=lambda x: x.strip()))
    biogen_minus_list.append(pd.read_table(biogen_minus[inp]).rename(columns=lambda x: x.strip()))
    conductance_minus_list.append(pd.read_table(conductance_minus[inp]).rename(columns=lambda x: x.strip()))
    ESTM_minus_list.append(pd.read_table(ESTM_minus[inp]).rename(columns=lambda x: x.strip()))
    irrigation_minus_list.append(pd.read_table(irrigation_minus[inp]).rename(columns=lambda x: x.strip()))
    OHM_minus_list.append(pd.read_table(OHM_minus[inp]).rename(columns=lambda x: x.strip()))
    Profiles_minus_list.append(pd.read_table(Profiles_minus[inp]).rename(columns=lambda x: x.strip()))
    soil_char_minus_list.append(pd.read_table(soil_char_minus[inp]).rename(columns=lambda x: x.strip()))
    surface_reflectance_minus_list.append(pd.read_table(surface_reflectance_minus[inp]).rename(columns=lambda x: x.strip()))
    surface_water_minus_list.append(pd.read_table(surface_water_minus[inp]).rename(columns=lambda x: x.strip()))
    Veg_minus_list.append(pd.read_table(Veg_minus[inp]).rename(columns=lambda x: x.strip()))
    
    anthro_plus_list.append(pd.read_table(anthro_plus[inp]).rename(columns=lambda x: x.strip()))
    biogen_plus_list.append(pd.read_table(biogen_plus[inp]).rename(columns=lambda x: x.strip()))
    conductance_plus_list.append(pd.read_table(conductance_plus[inp]).rename(columns=lambda x: x.strip()))
    ESTM_plus_list.append(pd.read_table(ESTM_plus[inp]).rename(columns=lambda x: x.strip()))
    irrigation_plus_list.append(pd.read_table(irrigation_plus[inp]).rename(columns=lambda x: x.strip()))
    OHM_plus_list.append(pd.read_table(OHM_plus[inp]).rename(columns=lambda x: x.strip()))
    Profiles_plus_list.append(pd.read_table(Profiles_plus[inp]).rename(columns=lambda x: x.strip()))
    soil_char_plus_list.append(pd.read_table(soil_char_plus[inp]).rename(columns=lambda x: x.strip()))
    surface_reflectance_plus_list.append(pd.read_table(surface_reflectance_plus[inp]).rename(columns=lambda x: x.strip()))
    surface_water_plus_list.append(pd.read_table(surface_water_plus[inp]).rename(columns=lambda x: x.strip()))
    Veg_plus_list.append(pd.read_table(Veg_plus[inp]).rename(columns=lambda x: x.strip()))



# %%
origAll = pd.concat(original_list)
cols_SA = ["Tsurf","QS","QH","QE","Evap","RO","TotCh","Drainage","SMD"]
origMax = origAll[cols_SA].max()
origMin = origAll[cols_SA].min()
normDen = origMax-origMin

# %%
def scoreForSensitivity(df_sense_list, orig_list, cols):
    origAll = pd.concat(orig_list)
    origMax = origAll[cols].max()
    origMin = origAll[cols].min()
    normDen = origMax-origMin

    df_sense = pd.concat(df_sense_list)
    heat_scoring = (df_sense["QE"]-origMin["QE"])/normDen["QE"] + (df_sense["Evap"]-origMin["Evap"])/normDen["Evap"] - ((df_sense["QS"]-df_sense["QS"])/normDen["QS"] + (df_sense["QH"]-origMin["QH"])/normDen["QH"] + (df_sense["Tsurf"]-origMin["Tsurf"])/normDen["Tsurf"])
    water_scoring = (df_sense["TotCh"]-origMin["TotCh"])/normDen["TotCh"] - ((df_sense["SMD"]-df_sense["SMD"])/normDen["SMD"] + (df_sense["Evap"]-origMin["Evap"])/normDen["Evap"] + (df_sense["RO"]-origMin["RO"])/normDen["RO"])
    heat_scoring = heat_scoring.groupby(heat_scoring.index).mean()
    water_scoring = water_scoring.groupby(water_scoring.index).mean()
    df_sense_avg = df_sense.groupby(df_sense.index).mean()
    df_sense_avg["heat_score"] = heat_scoring
    df_sense_avg["water_score"] = water_scoring

    return df_sense_avg

# %%
#anthro
anthro_minus_df = scoreForSensitivity(anthro_minus_list,original_list, cols_SA) 
anthro_plus_df = scoreForSensitivity(anthro_plus_list,original_list, cols_SA) 

#biogen
biogen_minus_df = scoreForSensitivity(biogen_minus_list,original_list, cols_SA) 
biogen_plus_df = scoreForSensitivity(biogen_plus_list,original_list, cols_SA) 

#conductance
conductance_minus_df = scoreForSensitivity(conductance_minus_list,original_list, cols_SA) 
conductance_plus_df = scoreForSensitivity(conductance_plus_list,original_list, cols_SA)

#ESTM
ESTM_minus_df = scoreForSensitivity(ESTM_minus_list,original_list, cols_SA) 
ESTM_plus_df = scoreForSensitivity(ESTM_plus_list,original_list, cols_SA) 

#irrigation
irrigation_minus_df = scoreForSensitivity(irrigation_minus_list,original_list, cols_SA) 
irrigation_plus_df = scoreForSensitivity(irrigation_plus_list,original_list, cols_SA) 

#OHM
OHM_minus_df = scoreForSensitivity(OHM_minus_list,original_list, cols_SA) 
OHM_plus_df = scoreForSensitivity(OHM_plus_list,original_list, cols_SA) 

#Profiles
Profiles_minus_df = scoreForSensitivity(Profiles_minus_list,original_list, cols_SA) 
Profiles_plus_df = scoreForSensitivity(Profiles_plus_list,original_list, cols_SA) 

#soil_char
soil_char_minus_df = scoreForSensitivity(soil_char_minus_list,original_list, cols_SA) 
soil_char_plus_df = scoreForSensitivity(soil_char_plus_list,original_list, cols_SA) 

#surface_reflectance
surface_reflectance_minus_df = scoreForSensitivity(surface_reflectance_minus_list,original_list, cols_SA) 
surface_reflectance_plus_df = scoreForSensitivity(surface_reflectance_plus_list,original_list, cols_SA) 

#surface_water
surface_water_minus_df = scoreForSensitivity(surface_water_minus_list,original_list, cols_SA) 
surface_water_plus_df = scoreForSensitivity(surface_water_plus_list,original_list, cols_SA) 

#Veg
Veg_minus_df = scoreForSensitivity(Veg_minus_list,original_list, cols_SA) 
Veg_plus_df = scoreForSensitivity(Veg_plus_list,original_list, cols_SA) 


# %%
Veg_plus_df

# %%
original_score = scoreForSensitivity(original_list,original_list, cols_SA)

# %%
def SA_calc_v1(df_list, orig_df, col_names):
    minus_tile = df_list[0].copy()
    plus_tile = df_list[1].copy()
    sensitivity_minus = minus_tile[col_names] - orig_df[col_names]
    sensitivity_plus = plus_tile[col_names] - orig_df[col_names]
    sensitivity = [sensitivity_minus, sensitivity_plus]
    return sensitivity

# %%
def df_moving_average(df, avg_df, inp):
    if inp == 0:
        return_df = df
    else:
        return_df = (avg_df + df)*0.5
    return return_df

# %%
def SA_calc_v2(df_list, orig_df, sense_vars, col_names, normalized=True):
    minus_tile = df_list[0].copy()
    plus_tile = df_list[1].copy()
    sensitivity_minus = minus_tile[col_names] - orig_df[col_names]
    sensitivity_plus = plus_tile[col_names] - orig_df[col_names]

    input_vars = sense_vars.loc["orig"]
    minus_diff_vars = sense_vars.loc["minus_diff"]
    plus_diff_vars = sense_vars.loc["plus_diff"]

    vars = iter(range(len(input_vars)))

    sense_minus_df = pd.DataFrame()
    sense_plus_df = pd.DataFrame()

    orig_mean = orig_df[col_names].mean()

    for var in vars:
        if (minus_diff_vars[var] == 0) or (plus_diff_vars[var] == 0):
            continue
        if normalized:
            sense_minus = sensitivity_minus/minus_diff_vars[var]
            sense_plus = sensitivity_plus/plus_diff_vars[var]

            norm_sense_minus = sense_minus*input_vars[var]/orig_mean
            norm_sense_plus = sense_plus*input_vars[var]/orig_mean

            sense_minus_df = df_moving_average(norm_sense_minus,sense_minus_df, var)
            sense_plus_df = df_moving_average(norm_sense_plus,sense_plus_df, var)
        else:
            sense_minus = sensitivity_minus/minus_diff_vars[var]
            sense_plus = sensitivity_plus/plus_diff_vars[var]
            sense_minus_df = df_moving_average(sense_minus,sense_minus_df, var)
            sense_plus_df = df_moving_average(sense_plus,sense_plus_df, var)

    sensitivity = [sense_minus_df, sense_plus_df]
    return sensitivity

# %%
sens_labels = ["anthro-","anthro+","biogen-","biogen+","conductance-","conductance+",
               "ESTM-","ESTM+","irrigation-","irrigation+","OHM-","OHM+","Profiles-",
               "Profiles+","soil_char-","soil_char+","surface_reflectance-",
               "surface_reflectance+","surface_water-","surface_water+","Veg-","Veg+"]

# %%
sens_labels_only = ["anthro","biogen","conductance", "ESTM","irrigation","OHM",
                    "Profiles","soil_char","surface_reflectance","surface_water","Veg"]

# %%
SA_inputs_list_v2 = [SA_inputs_list[0], pd.DataFrame(), SA_inputs_list[1], pd.DataFrame(),
                    pd.DataFrame(), SA_inputs_list[2], pd.DataFrame(), SA_inputs_list[3], 
                    SA_inputs_list[4], SA_inputs_list[5], SA_inputs_list[6]]

# %%
SA_inputs_dict = dict(zip(sens_labels_only,SA_inputs_list_v2))

# %%
sens_labels = ["heat_score_anthro-","water_score_anthro-","heat_score_anthro+","water_score_anthro+",
               "heat_score_biogen-","water_score_biogen-","heat_score_biogen+","water_score_biogen+",
               "heat_score_conductance-","water_score_conductance-","heat_score_conductance+","water_score_conductance+",
               "heat_score_ESTM-","water_score_ESTM-","heat_score_ESTM+","water_score_ESTM+",
               "heat_score_irrigation-","water_score_irrigation-","heat_score_irrigation+","water_score_irrigation+",
               "heat_score_OHM-","water_score_OHM-","heat_score_OHM+","water_score_OHM+",
               "heat_score_Profiles-","water_score_Profiles-", "heat_score_Profiles+","water_score_Profiles+",
               "heat_score_soil_char-","water_score_soil_char-","heat_score_soil_char+","water_score_soil_char+",
               "heat_score_surface_reflectance-","water_score_surface_reflectance-", 
               "heat_score_surface_reflectance+", "water_score_surface_reflectance+",
               "heat_score_surface_water-", "water_score_surface_water-","heat_score_surface_water+","water_score_surface_water+",
               "heat_score_Veg-","water_score_Veg-","heat_score_Veg+","water_score_Veg+"]

# %%
sensitivity_check_df =pd.concat([anthro_minus_df[["heat_score","water_score"]],
                                         anthro_plus_df[["heat_score","water_score"]],
                                         biogen_minus_df[["heat_score","water_score"]],
                                         biogen_plus_df[["heat_score","water_score"]],
                                         conductance_minus_df[["heat_score","water_score"]],
                                         conductance_plus_df[["heat_score","water_score"]],
                                         ESTM_minus_df[["heat_score","water_score"]],
                                         ESTM_plus_df[["heat_score","water_score"]], irrigation_minus_df[["heat_score","water_score"]],
                                         irrigation_plus_df[["heat_score","water_score"]],
                                         OHM_minus_df[["heat_score","water_score"]],
                                         OHM_plus_df[["heat_score","water_score"]],
                                         Profiles_minus_df[["heat_score","water_score"]],
                                         Profiles_plus_df[["heat_score","water_score"]], soil_char_minus_df[["heat_score","water_score"]],
                                         soil_char_plus_df[["heat_score","water_score"]],
                                         surface_reflectance_minus_df[["heat_score","water_score"]],
                                         surface_reflectance_plus_df[["heat_score","water_score"]],
                                         surface_water_minus_df[["heat_score","water_score"]], surface_water_plus_df[["heat_score","water_score"]],
                                         Veg_minus_df[["heat_score","water_score"]],Veg_plus_df[["heat_score","water_score"]]], axis=1)
sensitivity_check_df.columns = sens_labels

# %%
def SA_calc_Scores(df, orig_df, var_name, sense_vars=None, normalized=False):
    heat_var_min = 'heat_score_'+var_name+'-'
    water_var_min = 'water_score_'+var_name+'-'
    heat_var_plus = 'heat_score_'+var_name+'+'
    water_var_plus = 'water_score_'+var_name+'+'
    minus_heat = df[heat_var_min].copy()
    minus_water = df[water_var_min].copy()
    plus_heat = df[heat_var_plus].copy()
    plus_water = df[water_var_plus].copy()
    sensitivity_heat_minus = minus_heat - orig_df['heat_score']
    sensitivity_water_minus = minus_water - orig_df['water_score']
    sensitivity_heat_plus = plus_heat - orig_df['heat_score']
    sensitivity_water_plus = plus_water - orig_df['water_score']

    senseHeatMinVar = 'sense_'+heat_var_min
    senseWaterMinVar = 'sense_'+water_var_min
    senseHeatPlusVar = 'sense_'+heat_var_plus
    senseWaterPlusVar = 'sense_'+water_var_plus
    df[senseHeatMinVar] = sensitivity_heat_minus
    df[senseWaterMinVar] = sensitivity_water_minus
    df[senseHeatPlusVar] = sensitivity_heat_plus
    df[senseWaterPlusVar] = sensitivity_water_plus

    if normalized:
        input_vars = sense_vars.loc["orig"]
        minus_diff_vars = sense_vars.loc["minus_diff"]
        minus_diff_vars = minus_diff_vars.loc[minus_diff_vars != 0]
        plus_diff_vars = sense_vars.loc["plus_diff"]
        plus_diff_vars = plus_diff_vars.loc[plus_diff_vars != 0]

        orig_heat_mean = np.mean(orig_df['heat_score'])
        orig_water_mean = np.mean(orig_df['water_score'])

        min_heat_ratio = np.mean(input_vars/minus_diff_vars)/orig_heat_mean
        min_water_ratio = np.mean(input_vars/minus_diff_vars)/orig_water_mean
        plus_heat_ratio = np.mean(input_vars/plus_diff_vars)/orig_heat_mean
        plus_water_ratio = np.mean(input_vars/plus_diff_vars)/orig_water_mean

        normSenseHeatMinus = sensitivity_heat_minus * min_heat_ratio
        normSenseWaterMinus = sensitivity_water_minus * min_water_ratio
        normSenseHeatPlus = sensitivity_heat_plus * plus_heat_ratio
        normSenseWaterPlus = sensitivity_water_plus * plus_water_ratio

        normSenseHeatMinVar = 'norm_sense_'+heat_var_min
        normSenseWaterMinVar = 'norm_sense_'+water_var_min
        normSenseHeatPlusVar = 'norm_sense_'+heat_var_plus
        normSenseWaterPlusVar = 'norm_sense_'+water_var_plus
        df[normSenseHeatMinVar] = normSenseHeatMinus
        df[normSenseWaterMinVar] = normSenseWaterMinus
        df[normSenseHeatPlusVar] = normSenseHeatPlus
        df[normSenseWaterPlusVar] = normSenseWaterPlus

    return df

# %%
sensitivity_df = sensitivity_check_df.copy()
for sense in sens_labels_only:
    if SA_inputs_dict[sense].empty:
        sensitivity_df = SA_calc_Scores(sensitivity_df, original_score, sense)
    else:
        sensitivity_df = SA_calc_Scores(sensitivity_df, original_score, sense, SA_inputs_dict[sense], normalized=True)


# %%
sensitivity_df

# %%
sense_heat = sensitivity_df.filter(regex="norm_sense_heat").copy()
sense_water = sensitivity_df.filter(regex="norm_sense_water").copy()

# %%
sense_heat_plus = sense_heat.filter(regex="\+").copy()
sense_heat_min = sense_heat.filter(regex="-").copy()
sense_water_min = sense_water.filter(regex="-").copy()
sense_water_plus = sense_water.filter(regex="\+").copy()

# %%
sense_heat_plus

# %%
# SA version 1
saInputs = iter(range(len(original)))

anthro_sensitivity_list = []
biogen_sensitivity_list = []
conductance_sensitivity_list = []
ESTM_sensitivity_list = []
irrigation_sensitivity_list = []
OHM_sensitivity_list = []
Profiles_sensitivity_list = []
soil_char_sensitivity_list = []
surface_reflectance_sensitivity_list = []
surface_water_sensitivity_list = []
Veg_sensitivity_list = []

for inp in saInputs:
    orig_tile = original_list[inp].copy()
    anthro_sensitivity_list.append(SA_calc_v1(anthro_list[inp], orig_tile, cols_SA))
    biogen_sensitivity_list.append(SA_calc_v1(biogen_list[inp], orig_tile, cols_SA))
    conductance_sensitivity_list.append(SA_calc_v1(conductance_list[inp], orig_tile, cols_SA))
    ESTM_sensitivity_list.append(SA_calc_v1(ESTM_list[inp], orig_tile, cols_SA))
    irrigation_sensitivity_list.append(SA_calc_v1(irrigation_list[inp], orig_tile, cols_SA))
    OHM_sensitivity_list.append(SA_calc_v1(OHM_list[inp], orig_tile, cols_SA))
    Profiles_sensitivity_list.append(SA_calc_v1(Profiles_list[inp], orig_tile, cols_SA))
    soil_char_sensitivity_list.append(SA_calc_v1(soil_char_list[inp], orig_tile, cols_SA))
    surface_reflectance_sensitivity_list.append(SA_calc_v1(surface_reflectance_list[inp], orig_tile, cols_SA))
    surface_water_sensitivity_list.append(SA_calc_v1(surface_water_list[inp], orig_tile, cols_SA))
    Veg_sensitivity_list.append(SA_calc_v1(Veg_list[inp], orig_tile, cols_SA))

# %%
# SA version 2
saInputs = iter(range(len(original)))

anthro_sensitivity_list = []
biogen_sensitivity_list = []
conductance_sensitivity_list = []
ESTM_sensitivity_list = []
irrigation_sensitivity_list = []
OHM_sensitivity_list = []
Profiles_sensitivity_list = []
soil_char_sensitivity_list = []
surface_reflectance_sensitivity_list = []
surface_water_sensitivity_list = []
Veg_sensitivity_list = []

for inp in saInputs:
    orig_tile = original_list[inp].copy()
    anthro_sensitivity_list.append(SA_calc_v2(anthro_list[inp], orig_tile, anthro_inputs, cols_SA))
    biogen_sensitivity_list.append(SA_calc_v1(biogen_list[inp], orig_tile, cols_SA))
    conductance_sensitivity_list.append(SA_calc_v2(conductance_list[inp], orig_tile, conductance_inputs, cols_SA))
    ESTM_sensitivity_list.append(SA_calc_v1(ESTM_list[inp], orig_tile, cols_SA))
    irrigation_sensitivity_list.append(SA_calc_v1(irrigation_list[inp], orig_tile, cols_SA))
    OHM_sensitivity_list.append(SA_calc_v2(OHM_list[inp], orig_tile, OHM_inputs, cols_SA))
    Profiles_sensitivity_list.append(SA_calc_v1(Profiles_list[inp], orig_tile, cols_SA))
    soil_char_sensitivity_list.append(SA_calc_v2(soil_char_list[inp], orig_tile, soil_char_inputs, cols_SA))
    surface_reflectance_sensitivity_list.append(SA_calc_v2(surface_reflectance_list[inp], orig_tile, surface_reflectance_inputs, cols_SA))
    surface_water_sensitivity_list.append(SA_calc_v2(surface_water_list[inp], orig_tile, surface_water_inputs, cols_SA))
    Veg_sensitivity_list.append(SA_calc_v2(Veg_list[inp], orig_tile, Veg_inputs, cols_SA))

# %% [markdown]
# Need to average it across all grid cells.

# %%
saInputs = iter(range(len(original)))

anthro_minus = pd.DataFrame()
anthro_plus = pd.DataFrame()
biogen_minus = pd.DataFrame()
biogen_plus = pd.DataFrame()
conductance_minus = pd.DataFrame()
conductance_plus = pd.DataFrame()
ESTM_minus = pd.DataFrame()
ESTM_plus = pd.DataFrame()
irrigation_minus = pd.DataFrame()
irrigation_plus = pd.DataFrame()
OHM_minus = pd.DataFrame()
OHM_plus = pd.DataFrame()
Profiles_minus = pd.DataFrame()
Profiles_plus = pd.DataFrame()
soil_char_minus = pd.DataFrame()
soil_char_plus = pd.DataFrame()
surface_reflectance_minus = pd.DataFrame()
surface_reflectance_plus = pd.DataFrame()
surface_water_minus = pd.DataFrame()
surface_water_plus = pd.DataFrame()
Veg_minus = pd.DataFrame()
Veg_plus = pd.DataFrame()

for inp in saInputs:
    anthro_minus = df_moving_average(anthro_sensitivity_list[inp][0], anthro_minus,inp)
    anthro_plus = df_moving_average(anthro_sensitivity_list[inp][1], anthro_plus, inp)
    biogen_minus = df_moving_average(biogen_sensitivity_list[inp][0], biogen_minus, inp)
    biogen_plus = df_moving_average(biogen_sensitivity_list[inp][1], biogen_plus, inp)
    conductance_minus = df_moving_average(conductance_sensitivity_list[inp][0], 
                                          conductance_minus, inp)
    conductance_plus = df_moving_average(conductance_sensitivity_list[inp][1], 
                                         conductance_plus, inp)
    ESTM_minus = df_moving_average(ESTM_sensitivity_list[inp][0], ESTM_minus, inp)
    ESTM_plus = df_moving_average(ESTM_sensitivity_list[inp][1], ESTM_plus, inp)
    irrigation_minus = df_moving_average(irrigation_sensitivity_list[inp][0], irrigation_minus, inp)
    irrigation_plus = df_moving_average(irrigation_sensitivity_list[inp][1], irrigation_plus, inp)
    OHM_minus = df_moving_average(OHM_sensitivity_list[inp][0], OHM_minus, inp)
    OHM_plus = df_moving_average(OHM_sensitivity_list[inp][1], OHM_plus, inp)
    Profiles_minus = df_moving_average(Profiles_sensitivity_list[inp][0], Profiles_minus, inp)
    Profiles_plus = df_moving_average(Profiles_sensitivity_list[inp][1], Profiles_plus, inp)
    soil_char_minus = df_moving_average(soil_char_sensitivity_list[inp][0], soil_char_minus, inp)
    soil_char_plus = df_moving_average(soil_char_sensitivity_list[inp][1], soil_char_plus, inp)
    surface_reflectance_minus = df_moving_average(surface_reflectance_sensitivity_list[inp][0], 
                                                  surface_reflectance_minus, inp)
    surface_reflectance_plus = df_moving_average(surface_reflectance_sensitivity_list[inp][1], 
                                                 surface_reflectance_plus, inp)
    surface_water_minus = df_moving_average(surface_water_sensitivity_list[inp][0], 
                                            surface_water_minus, inp)
    surface_water_plus = df_moving_average(surface_water_sensitivity_list[inp][1], 
                                           surface_water_plus, inp)
    Veg_minus = df_moving_average(Veg_sensitivity_list[inp][0], Veg_minus, inp)
    Veg_plus = df_moving_average(Veg_sensitivity_list[inp][1], Veg_plus, inp)


# %% [markdown]
# anthro_plus

# %%
sens_labels = ["anthro-","anthro+","biogen-","biogen+","conductance-","conductance+",
               "ESTM-","ESTM+","irrigation-","irrigation+","OHM-","OHM+","Profiles-",
               "Profiles+","soil_char-","soil_char+","surface_reflectance-",
               "surface_reflectance+","surface_water-","surface_water+","Veg-","Veg+"]

# %%
Tsurf_SA_df = pd.DataFrame([anthro_minus["Tsurf"],anthro_plus["Tsurf"],biogen_minus["Tsurf"],
                           biogen_plus["Tsurf"],conductance_minus["Tsurf"],
                           conductance_plus["Tsurf"],ESTM_minus["Tsurf"],ESTM_plus["Tsurf"],
                           irrigation_minus["Tsurf"],irrigation_plus["Tsurf"],OHM_minus["Tsurf"],
                           OHM_plus["Tsurf"],Profiles_minus["Tsurf"],Profiles_plus["Tsurf"],
                           soil_char_minus["Tsurf"],soil_char_plus["Tsurf"],
                           surface_reflectance_minus["Tsurf"],surface_reflectance_plus["Tsurf"],
                           surface_water_minus["Tsurf"],surface_water_plus["Tsurf"],
                           Veg_minus["Tsurf"],Veg_plus["Tsurf"]],index=sens_labels).transpose()

QS_SA_df = pd.DataFrame([anthro_minus["QS"],anthro_plus["QS"],biogen_minus["QS"],
                        biogen_plus["QS"],conductance_minus["QS"],
                        conductance_plus["QS"],ESTM_minus["QS"],ESTM_plus["QS"],
                        irrigation_minus["QS"],irrigation_plus["QS"],OHM_minus["QS"],
                        OHM_plus["QS"],Profiles_minus["QS"],Profiles_plus["QS"],
                        soil_char_minus["QS"],soil_char_plus["QS"],
                        surface_reflectance_minus["QS"],surface_reflectance_plus["QS"],
                        surface_water_minus["QS"],surface_water_plus["QS"],
                        Veg_minus["QS"],Veg_plus["QS"]],index=sens_labels).transpose()

QH_SA_df = pd.DataFrame([anthro_minus["QH"],anthro_plus["QH"],biogen_minus["QH"],
                        biogen_plus["QH"],conductance_minus["QH"],
                        conductance_plus["QH"],ESTM_minus["QH"],ESTM_plus["QH"],
                        irrigation_minus["QH"],irrigation_plus["QH"],OHM_minus["QH"],
                        OHM_plus["QH"],Profiles_minus["QH"],Profiles_plus["QH"],
                        soil_char_minus["QH"],soil_char_plus["QH"],
                        surface_reflectance_minus["QH"],surface_reflectance_plus["QH"],
                        surface_water_minus["QH"],surface_water_plus["QH"],
                        Veg_minus["QH"],Veg_plus["QH"]],index=sens_labels).transpose()

QE_SA_df = pd.DataFrame([anthro_minus["QE"],anthro_plus["QE"],biogen_minus["QE"],
                        biogen_plus["QE"],conductance_minus["QE"],
                        conductance_plus["QE"],ESTM_minus["QE"],ESTM_plus["QE"],
                        irrigation_minus["QE"],irrigation_plus["QE"],OHM_minus["QE"],
                        OHM_plus["QE"],Profiles_minus["QE"],Profiles_plus["QE"],
                        soil_char_minus["QE"],soil_char_plus["QE"],
                        surface_reflectance_minus["QE"],surface_reflectance_plus["QE"],
                        surface_water_minus["QE"],surface_water_plus["QE"],
                        Veg_minus["QE"],Veg_plus["QE"]],index=sens_labels).transpose()

Evap_SA_df = pd.DataFrame([anthro_minus["Evap"],anthro_plus["Evap"],biogen_minus["Evap"],
                           biogen_plus["Evap"],conductance_minus["Evap"],
                           conductance_plus["Evap"],ESTM_minus["Evap"],ESTM_plus["Evap"],
                           irrigation_minus["Evap"],irrigation_plus["Evap"],OHM_minus["Evap"],
                           OHM_plus["Evap"],Profiles_minus["Evap"],Profiles_plus["Evap"],
                           soil_char_minus["Evap"],soil_char_plus["Evap"],
                           surface_reflectance_minus["Evap"],surface_reflectance_plus["Evap"],
                           surface_water_minus["Evap"],surface_water_plus["Evap"],
                           Veg_minus["Evap"],Veg_plus["Evap"]],index=sens_labels).transpose()

RO_SA_df = pd.DataFrame([anthro_minus["RO"],anthro_plus["RO"],biogen_minus["RO"],
                        biogen_plus["RO"],conductance_minus["RO"],
                        conductance_plus["RO"],ESTM_minus["RO"],ESTM_plus["RO"],
                        irrigation_minus["RO"],irrigation_plus["RO"],OHM_minus["RO"],
                        OHM_plus["RO"],Profiles_minus["RO"],Profiles_plus["RO"],
                        soil_char_minus["RO"],soil_char_plus["RO"],
                        surface_reflectance_minus["RO"],surface_reflectance_plus["RO"],
                        surface_water_minus["RO"],surface_water_plus["RO"],
                        Veg_minus["RO"],Veg_plus["RO"]],index=sens_labels).transpose()

TotCh_SA_df = pd.DataFrame([anthro_minus["TotCh"],anthro_plus["TotCh"],biogen_minus["TotCh"],
                            biogen_plus["TotCh"],conductance_minus["TotCh"],
                            conductance_plus["TotCh"],ESTM_minus["TotCh"],ESTM_plus["TotCh"],
                            irrigation_minus["TotCh"],irrigation_plus["TotCh"],OHM_minus["TotCh"],
                            OHM_plus["TotCh"],Profiles_minus["TotCh"],Profiles_plus["TotCh"],
                            soil_char_minus["TotCh"],soil_char_plus["TotCh"],
                            surface_reflectance_minus["TotCh"],surface_reflectance_plus["TotCh"],
                            surface_water_minus["TotCh"],surface_water_plus["TotCh"],
                            Veg_minus["TotCh"],Veg_plus["TotCh"]],index=sens_labels).transpose()

Drainage_SA_df = pd.DataFrame([anthro_minus["Drainage"],anthro_plus["Drainage"],
                               biogen_minus["Drainage"],biogen_plus["Drainage"],
                               conductance_minus["Drainage"],conductance_plus["Drainage"],
                               ESTM_minus["Drainage"],ESTM_plus["Drainage"], 
                               irrigation_minus["Drainage"],irrigation_plus["Drainage"],
                               OHM_minus["Drainage"],OHM_plus["Drainage"],
                               Profiles_minus["Drainage"],Profiles_plus["Drainage"],
                               soil_char_minus["Drainage"],soil_char_plus["Drainage"],
                               surface_reflectance_minus["Drainage"],
                               surface_reflectance_plus["Drainage"],
                               surface_water_minus["Drainage"],surface_water_plus["Drainage"],
                               Veg_minus["Drainage"],Veg_plus["Drainage"]],index=sens_labels).transpose()

SMD_SA_df = pd.DataFrame([anthro_minus["SMD"],anthro_plus["SMD"],biogen_minus["SMD"],
                          biogen_plus["SMD"],conductance_minus["SMD"],
                          conductance_plus["SMD"],ESTM_minus["SMD"],ESTM_plus["SMD"],
                          irrigation_minus["SMD"],irrigation_plus["SMD"],OHM_minus["SMD"],
                          OHM_plus["SMD"],Profiles_minus["SMD"],Profiles_plus["SMD"],
                          soil_char_minus["SMD"],soil_char_plus["SMD"],
                          surface_reflectance_minus["SMD"],surface_reflectance_plus["SMD"],
                          surface_water_minus["SMD"],surface_water_plus["SMD"],
                          Veg_minus["SMD"],Veg_plus["SMD"]],index=sens_labels).transpose()


# %%
def dailyAvgCalc(df, num_days=365, num_per_Day=1):
    days = iter(range(num_days))
    numHours = int(24/num_per_Day)
    daily_avg_list = []
    for day in days:
        numD = iter(range(num_per_Day))
        for num in numD:
            curr_day_start_hour = day*24 + int(num*numHours)
            curr_day_end_hour = curr_day_start_hour + numHours - 1
            curr_day_mean = df.iloc[df.index[curr_day_start_hour]:df.index[curr_day_end_hour]].mean()
            daily_avg_list.append(curr_day_mean)

    daily_avg_df = pd.DataFrame(daily_avg_list)
    return daily_avg_df


# %%
Tsurf_Daily_SA_df = dailyAvgCalc(Tsurf_SA_df)
QS_Daily_SA_df = dailyAvgCalc(QS_SA_df)
QH_Daily_SA_df = dailyAvgCalc(QH_SA_df)
QE_Daily_SA_df = dailyAvgCalc(QE_SA_df)
Evap_Daily_SA_df = dailyAvgCalc(Evap_SA_df)
RO_Daily_SA_df = dailyAvgCalc(RO_SA_df)
TotCh_Daily_SA_df = dailyAvgCalc(TotCh_SA_df)
Drainage_Daily_SA_df = dailyAvgCalc(Drainage_SA_df)
SMD_Daily_SA_df = dailyAvgCalc(SMD_SA_df)

# %%
sense_heat_min_daily = dailyAvgCalc(sense_heat_min)
sense_heat_plus_daily = dailyAvgCalc(sense_heat_plus)
sense_water_min_daily = dailyAvgCalc(sense_water_min)
sense_water_plus_daily = dailyAvgCalc(sense_water_plus)

sense_heat_min_dailyQ = dailyAvgCalc(sense_heat_min, num_per_Day=4)
sense_heat_plus_dailyQ = dailyAvgCalc(sense_heat_plus, num_per_Day=4)
sense_water_min_dailyQ = dailyAvgCalc(sense_water_min, num_per_Day=4)
sense_water_plus_dailyQ = dailyAvgCalc(sense_water_plus, num_per_Day=4)


norm_sense_heat_plus_daily = sense_heat_plus_daily/np.max([np.abs(sense_heat_plus_daily),
                                                          np.abs(sense_heat_min_daily)])
norm_sense_heat_min_daily = sense_heat_min_daily/np.max([np.abs(sense_heat_plus_daily),
                                                         np.abs(sense_heat_min_daily)])
norm_sense_water_min_daily = sense_water_min_daily/np.max([np.abs(sense_water_min_daily),
                                                           np.abs(sense_water_plus_daily)])
norm_sense_water_plus_daily = sense_water_plus_daily/np.max([np.abs(sense_water_min_daily),
                                                             np.abs(sense_water_plus_daily)])

norm_sense_heat_plus_dailyQ = sense_heat_min_dailyQ/np.max([np.abs(sense_heat_min_dailyQ),
                                                          np.abs(sense_heat_min_dailyQ)])
norm_sense_heat_min_dailyQ = sense_heat_min_dailyQ/np.max([np.abs(sense_heat_plus_dailyQ),
                                                         np.abs(sense_heat_min_dailyQ)])
norm_sense_water_min_dailyQ = sense_water_min_dailyQ/np.max([np.abs(sense_water_min_dailyQ),
                                                           np.abs(sense_water_plus_dailyQ)])
norm_sense_water_plus_dailyQ = sense_water_plus_daily/np.max([np.abs(sense_water_min_dailyQ),
                                                             np.abs(sense_water_plus_dailyQ)])

# %%
norm_sense_water_plus_daily

# %%
norm_sense_heat_plus_daily

# %%
normMin_diff = norm_sense_heat_min_daily - norm_sense_water_min_daily
normPlus_diff = norm_sense_heat_plus_daily - norm_sense_water_plus_daily

# %%
colors = ['m', 'c', 'r', 'k', 'y', 'b', 'g']

normMin_diff.plot(title='Minus Sensitivities Difference Between Heat and Water Scores', 
                         xlim=[0,365], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
colors = ['m', 'c', 'r', 'k', 'y', 'b', 'g']

normPlus_diff.plot(title='Plus Sensitivities Difference Between Heat and Water Scores', 
                         xlim=[0,365], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
norm_Tsurf_SA_df = Tsurf_SA_df/np.max([np.abs(np.max(Tsurf_SA_df)), 
                                       np.abs(np.min(Tsurf_SA_df))])
norm_QS_SA_df = QS_SA_df/np.max([np.abs(np.max(QS_SA_df)), np.abs(np.min(QS_SA_df))])
norm_QH_SA_df = QH_SA_df/np.max([np.abs(np.max(QH_SA_df)), np.abs(np.min(QH_SA_df))])
norm_QE_SA_df = QE_SA_df/np.max([np.abs(np.max(QE_SA_df)), np.abs(np.min(QE_SA_df))])
norm_Evap_SA_df = Evap_SA_df/np.max([np.abs(np.max(Evap_SA_df)), np.abs(np.min(Evap_SA_df))])
norm_RO_SA_df = RO_SA_df/np.max([np.abs(np.max(RO_SA_df)), np.abs(np.min(RO_SA_df))])
norm_TotCh_SA_df = TotCh_SA_df/np.max([np.abs(np.max(TotCh_SA_df)), 
                                       np.abs(np.min(TotCh_SA_df))])
norm_Drainage_SA_df = Drainage_SA_df/np.max([np.abs(np.max(Drainage_SA_df)), 
                                             np.abs(np.min(Drainage_SA_df))])
norm_SMD_SA_df = SMD_SA_df/np.max([np.abs(np.max(SMD_SA_df)), np.abs(np.min(SMD_SA_df))])

# %%
norm_sense_heat_plus = sense_heat_plus/np.max([np.abs(sense_heat_plus),
                                               np.abs(sense_heat_min)])
norm_sense_heat_min = sense_heat_min/np.max([np.abs(sense_heat_plus),
                                             np.abs(sense_heat_min)])
norm_sense_water_min = sense_water_min/np.max([np.abs(sense_water_min),
                                               np.abs(sense_water_plus)])
norm_sense_water_plus = sense_water_plus/np.max([np.abs(sense_water_min),
                                                 np.abs(sense_water_plus)])

# %%
sense_heat_min.max()


# %%
sense_heat_plus.max()


# %%
sense_water_min.max()


# %%
sense_water_plus.max()

# %%
SUEWS_labels = ['Bulk Surface Temperatures ($T_{surf}$)','Storage Heat Flux ($Q_{S}$)',
                'Sensible Heat Flux ($Q_{H}$)','Latent Heat Flux ($Q_{E}$)',
                'Evaporation Rate', 'Runoff Rate','Surface and Soil Moisture Change',
                'Drainage Rate', 'Soil Moisture Deficit']

average_sensitivities = pd.DataFrame([norm_Tsurf_SA_df.mean(),norm_QS_SA_df.mean(),
                                  norm_QH_SA_df.mean(),norm_QE_SA_df.mean(),
                                  norm_Evap_SA_df.mean(),norm_RO_SA_df.mean(),
                                  norm_TotCh_SA_df.mean(),norm_Drainage_SA_df.mean(),
                                  norm_SMD_SA_df.mean()])
average_sensitivities.index = SUEWS_labels

median_sensitivities = pd.DataFrame([norm_Tsurf_SA_df.median(),norm_QS_SA_df.median(),
                                    norm_QH_SA_df.median(),norm_QE_SA_df.median(),
                                    norm_Evap_SA_df.median(),norm_RO_SA_df.median(),
                                    norm_TotCh_SA_df.median(),norm_Drainage_SA_df.median(),
                                    norm_SMD_SA_df.median()])
median_sensitivities.index = SUEWS_labels


max_pos_sensitivities = pd.DataFrame([norm_Tsurf_SA_df.max(),norm_QS_SA_df.max(),
                                    norm_QH_SA_df.max(),norm_QE_SA_df.max(),
                                    norm_Evap_SA_df.max(),norm_RO_SA_df.max(),
                                    norm_TotCh_SA_df.max(),norm_Drainage_SA_df.max(),
                                    norm_SMD_SA_df.max()])
max_pos_sensitivities.index = SUEWS_labels

max_neg_sensitivities =pd.DataFrame([norm_Tsurf_SA_df.min(),norm_QS_SA_df.min(),
                                    norm_QH_SA_df.min(),norm_QE_SA_df.min(),
                                    norm_Evap_SA_df.min(),norm_RO_SA_df.min(),
                                    norm_TotCh_SA_df.min(),norm_Drainage_SA_df.min(),
                                    norm_SMD_SA_df.min()])
max_neg_sensitivities.index = SUEWS_labels


# %%
average_sensitivities.to_csv(folder + "average_sensitivities.csv")
median_sensitivities.to_csv(folder + "median_sensitivities.csv")
max_pos_sensitivities.to_csv(folder + "max_pos_sensitivities.csv")
max_neg_sensitivities.to_csv(folder + "max_neg_sensitivities.csv")

# %%
norm_Tsurf_SA_df.quantile([0, 1.0]) # only surface reflectance and vegetation impact this

# %%
norm_QS_SA_df.quantile([0, 1.0]) #OHM, surface reflectance, and veg


# %%
norm_QH_SA_df.quantile([0, 1.0]) # anthro, conductance, OHM, soil_char, 
                                 # surface reflectance, surface water, and veg


# %%
norm_QE_SA_df.quantile([0, 1.0])# anthro, conductance, OHM, soil_char, 
                                # surface reflectance, surface water, and veg


# %%
norm_Evap_SA_df.quantile([0, 1.0]) # anthro, conductance, OHM, soil_char, 
                                   # surface reflectance, surface water, and veg


# %%
norm_RO_SA_df.quantile([0, 1.0])# anthro, conductance, OHM, soil_char, 
                                # surface reflectance, surface water, and veg


# %%
norm_TotCh_SA_df.quantile([0, 1.0])# anthro, conductance, OHM, soil_char, 
                                   # surface reflectance, surface water, and veg


# %%
norm_Drainage_SA_df.quantile([0, 1.0]) # anthro, conductance, OHM, soil_char, 
                                       # surface reflectance, surface water, and veg


# %%
norm_SMD_SA_df.quantile([0, 1.0]) # anthro, conductance, OHM, soil_char, 
                                  # surface reflectance, surface water, and veg

# %%
norm_minus_Tsurf_SA_df = norm_Tsurf_SA_df[['surface_reflectance-', 'Veg-']]

norm_plus_Tsurf_SA_df = norm_Tsurf_SA_df[['surface_reflectance+', 'Veg+']]

norm_minus_QS_SA_df = norm_QS_SA_df[['OHM-', 'surface_reflectance-', 'Veg-']]

norm_plus_QS_SA_df = norm_QS_SA_df[['OHM+', 'surface_reflectance+', 'Veg+']]

norm_minus_QH_SA_df = norm_QH_SA_df[['anthro-', 'conductance-', 'OHM-', 'soil_char-',
                                     'surface_reflectance-', 'surface_water-', 'Veg-']]

norm_plus_QH_SA_df = norm_QH_SA_df[['anthro+','conductance+', 'OHM+', 'soil_char+', 
                                    'surface_reflectance+', 'surface_water+','Veg+']]

norm_minus_QE_SA_df = norm_QE_SA_df[['anthro-', 'conductance-', 'OHM-', 'soil_char-',
                                     'surface_reflectance-', 'surface_water-', 'Veg-']]

norm_plus_QE_SA_df = norm_QE_SA_df[['anthro+','conductance+', 'OHM+', 'soil_char+', 
                                    'surface_reflectance+', 'surface_water+','Veg+']]

norm_minus_Evap_SA_df = norm_Evap_SA_df[['anthro-', 'conductance-', 'OHM-', 'soil_char-',
                                         'surface_reflectance-', 'surface_water-', 'Veg-']]

norm_plus_Evap_SA_df = norm_Evap_SA_df[['anthro+','conductance+', 'OHM+', 'soil_char+',
                                        'surface_reflectance+', 'surface_water+','Veg+']]

norm_minus_RO_SA_df = norm_RO_SA_df[['anthro-', 'conductance-', 'OHM-', 'soil_char-',
                                     'surface_reflectance-', 'surface_water-', 'Veg-']]

norm_plus_RO_SA_df = norm_RO_SA_df[['anthro+','conductance+', 'OHM+', 'soil_char+', 
                                    'surface_reflectance+', 'surface_water+','Veg+']]

norm_minus_TotCh_SA_df = norm_TotCh_SA_df[['anthro-', 'conductance-', 'OHM-', 'soil_char-',
                                           'surface_reflectance-', 'surface_water-', 'Veg-']]

norm_plus_TotCh_SA_df = norm_TotCh_SA_df[['anthro+','conductance+', 'OHM+', 'soil_char+',
                                          'surface_reflectance+', 'surface_water+','Veg+']]

norm_minus_Drainage_SA_df = norm_Drainage_SA_df[['anthro-', 'conductance-', 'OHM-', 
                                                 'soil_char-', 'surface_reflectance-', 
                                                 'surface_water-', 'Veg-']]

norm_plus_Drainage_SA_df = norm_Drainage_SA_df[['anthro+','conductance+', 'OHM+', 
                                                'soil_char+', 'surface_reflectance+', 
                                                'surface_water+','Veg+']]

norm_minus_SMD_SA_df = norm_SMD_SA_df[['anthro-', 'conductance-', 'OHM-', 'soil_char-',
                                       'surface_reflectance-', 'surface_water-', 'Veg-']]

norm_plus_SMD_SA_df = norm_SMD_SA_df[['anthro+','conductance+', 'OHM+', 'soil_char+', 
                                      'surface_reflectance+', 'surface_water+','Veg+']]

# %%
norm_plus_SMD_SA_df

# %%

norm_minus_Tsurf_SA_df.index = norm_minus_Tsurf_SA_df.index/24
norm_plus_Tsurf_SA_df.index = norm_plus_Tsurf_SA_df.index/24

norm_minus_QS_SA_df.index = norm_minus_QS_SA_df.index/24
norm_plus_QS_SA_df.index = norm_plus_QS_SA_df.index/24

norm_minus_QH_SA_df.index = norm_minus_QH_SA_df.index/24
norm_plus_QH_SA_df.index = norm_plus_QH_SA_df.index/24

norm_minus_QE_SA_df .index = norm_minus_QE_SA_df.index/24
norm_plus_QE_SA_df.index = norm_plus_QE_SA_df.index/24

norm_minus_Evap_SA_df.index = norm_minus_Evap_SA_df.index/24
norm_plus_Evap_SA_df.index = norm_plus_Evap_SA_df.index/24

norm_minus_RO_SA_df.index = norm_minus_RO_SA_df.index/24
norm_plus_RO_SA_df.index = norm_plus_RO_SA_df.index/24

norm_minus_TotCh_SA_df.index = norm_minus_TotCh_SA_df.index/24
norm_plus_TotCh_SA_df.index = norm_plus_TotCh_SA_df.index/24

norm_minus_Drainage_SA_df.index = norm_minus_Drainage_SA_df.index/24
norm_plus_Drainage_SA_df.index = norm_plus_Drainage_SA_df.index/24

norm_minus_SMD_SA_df.index = norm_minus_SMD_SA_df.index/24
norm_plus_SMD_SA_df.index = norm_plus_SMD_SA_df.index/24

labels = ['Anthropogenic Heat','Surface Conductance', 'Objective Hysteresis Model', 
          'Sub-surface Soil Character', 'Surface Reflectance characteristics', 
          'Surface Water Storage characteristics','Vegetation characteristics']

norm_minus_Tsurf_SA_df.columns = ['Surface Reflectance characteristics', 
                                  'Vegetation characteristics']
norm_plus_Tsurf_SA_df.columns = ['Surface Reflectance characteristics', 
                                 'Vegetation characteristics']

norm_minus_QS_SA_df.columns =['Objective Hysteresis Model', 
                              'Surface Reflectance characteristics',
                              'Vegetation characteristics']
norm_plus_QS_SA_df.columns = ['Objective Hysteresis Model', 
                              'Surface Reflectance characteristics',
                              'Vegetation characteristics']

norm_minus_QH_SA_df.columns = labels
norm_plus_QH_SA_df.columns = labels
norm_minus_QE_SA_df.columns = labels
norm_plus_QE_SA_df.columns = labels
norm_minus_Evap_SA_df.columns = labels
norm_plus_Evap_SA_df.columns = labels
norm_minus_RO_SA_df.columns = labels
norm_plus_RO_SA_df.columns = labels
norm_minus_TotCh_SA_df.columns = labels
norm_plus_TotCh_SA_df.columns = labels
norm_minus_Drainage_SA_df.columns = labels
norm_plus_Drainage_SA_df.columns = labels
norm_minus_SMD_SA_df.columns = labels
norm_plus_SMD_SA_df.columns = labels


# %%
#most recent SA
norm_sense_heat_plus.index = norm_sense_heat_plus.index/24
norm_sense_heat_min.index = norm_sense_heat_min.index/24
norm_sense_water_min.index = norm_sense_water_min.index/24
norm_sense_water_plus.index = norm_sense_water_plus.index/24

labels = ['Anthropogenic Heat','Surface Conductance', 'Objective Hysteresis Model', 
          'Sub-surface Soil Character', 'Surface Reflectance characteristics', 
          'Surface Water Storage characteristics','Vegetation characteristics']

norm_sense_heat_plus.columns = labels
norm_sense_heat_min.columns = labels
norm_sense_water_min.columns = labels
norm_sense_water_plus.columns = labels

# %%
norm_sense_heat_min_daily.columns = labels
norm_sense_heat_plus_daily.columns = labels
norm_sense_water_min_daily.columns = labels
norm_sense_water_plus_daily.columns = labels

# %%
SUEWS_labels = ['Bulk Surface Temperatures ($T_{surf}$)','Storage Heat Flux ($Q_{S}$)',
                'Sensible Heat Flux ($Q_{H}$)','Latent Heat Flux ($Q_{E}$)',
                'Evaporation Rate', 'Runoff Rate','Surface and Soil Moisture Change',
                'Drainage Rate', 'Soil Moisture Deficit']

# %% [markdown]
# ##### Heat Score Sensitivity

# %%
colors = ['m', 'c', 'r', 'k', 'y', 'b', 'g']

norm_sense_heat_min.plot(title='Minus Sensitivities Heat Score', 
                         xlim=[0,365], ylim=[-1,1], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
colors = ['m', 'c', 'r', 'k', 'y', 'b', 'g']
plt.style.use('ggplot')
norm_sense_heat_min_daily.plot(title='Minus Sensitivities Heat Score Daily Average', 
                         xlim=[0,365], ylim=[-1,1], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
colors = ['m', 'c', 'r', 'k', 'y', 'b', 'g']

norm_sense_heat_plus.plot(title='Plus Sensitivities Heat Score', 
                         xlim=[0,365], ylim=[-1,1], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
colors = ['m', 'c', 'r', 'k', 'y', 'b', 'g']

norm_sense_heat_plus_daily.plot(title='Plus Sensitivities Heat Score Daily Average', 
                         xlim=[0,365], ylim=[-1,1], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
norm_sense_heat_plus_daily.columns

# %%
anthro_heat = pd.concat([norm_sense_heat_min_daily[["Anthropogenic Heat"]],norm_sense_heat_plus_daily[["Anthropogenic Heat"]]], axis=1)
anthro_heat.columns = ["Anthropogenic Heat Minus", "Anthropogenic Heat Plus"]
anthro_heat.plot(title='Anthro Sensitivities Heat Score', 
                         xlim=[0,365], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%

#        'Objective Hysteresis Model', 'Sub-surface Soil Character',
#        'Surface Reflectance characteristics',
#        'Surface Water Storage characteristics', 'Vegetation characteristics'],
#       dtype='object')
surf_cond_heat = pd.concat([norm_sense_heat_min_daily[["Surface Conductance"]],norm_sense_heat_plus_daily[["Surface Conductance"]]], axis=1)
surf_cond_heat.columns = ["Surface Conductance Minus", "Surface Conductance Plus"]
surf_cond_heat.plot(title='Surface Conductance Sensitivities Heat Score', 
                         xlim=[0,365], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
#        'Objective Hysteresis Model', 'Sub-surface Soil Character',
#        'Surface Reflectance characteristics',
#        'Surface Water Storage characteristics', 'Vegetation characteristics'],
#       dtype='object')
OHM_heat = pd.concat([norm_sense_heat_min_daily[["Objective Hysteresis Model"]],norm_sense_heat_plus_daily[["Objective Hysteresis Model"]]], axis=1)
OHM_heat.columns = ["Objective Hysteresis Model Minus", "Objective Hysteresis Model Plus"]
OHM_heat.plot(title='Objective Hysteresis Model Sensitivities Heat Score', 
                         xlim=[0,365], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
#        'Surface Reflectance characteristics',
#        'Surface Water Storage characteristics', 'Vegetation characteristics'],
#       dtype='object')
soil_heat = pd.concat([norm_sense_heat_min_daily[["Sub-surface Soil Character"]],norm_sense_heat_plus_daily[["Sub-surface Soil Character"]]], axis=1)
soil_heat.columns = ["Sub-surface Soil Character Minus", "Sub-surface Soil Character Plus"]
soil_heat.plot(title='Sub-surface Soil Character Sensitivities Heat Score', 
                         xlim=[0,365], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
#        'Surface Water Storage characteristics', 'Vegetation characteristics'],
#       dtype='object')
surf_Reflect_heat = pd.concat([norm_sense_heat_min_daily[["Surface Reflectance characteristics"]],norm_sense_heat_plus_daily[["Surface Reflectance characteristics"]]], axis=1)
surf_Reflect_heat.columns = ["Surface Reflectance characteristics Minus", "Surface Reflectance characteristics Plus"]
surf_Reflect_heat.plot(title='Surface Reflectance characteristics Sensitivities Heat Score', 
                         xlim=[0,365], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
surf_storage_heat = pd.concat([norm_sense_heat_min_daily[["Surface Water Storage characteristics"]],norm_sense_heat_plus_daily[["Surface Water Storage characteristics"]]], axis=1)
surf_storage_heat.columns = ["Surface Water Storage characteristics Minus", "Surface Water Storage characteristics Plus"]
surf_storage_heat.plot(title='Surface Water Storage characteristics Sensitivities Heat Score', 
                         xlim=[0,365], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
#      'Vegetation characteristics'],
#       dtype='object')
veg_heat = pd.concat([norm_sense_heat_min_daily[["Vegetation characteristics"]],norm_sense_heat_plus_daily[["Vegetation characteristics"]]], axis=1)
veg_heat.columns = ["Vegetation characteristics Minus", "Vegetation characteristics Plus"]
veg_heat.plot(title='Vegetation characteristics Sensitivities Heat Score', 
                         xlim=[0,365], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %% [markdown]
# ##### Water Score Sensitivity

# %%
colors = ['m', 'c', 'r', 'k', 'y', 'b', 'g']

norm_sense_water_min.plot(title='Minus Sensitivities Water Score', 
                         xlim=[0,365], ylim=[-1,1], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
colors = ['m', 'c', 'r', 'k', 'y', 'b', 'g']

norm_sense_water_min_daily.plot(title='Minus Sensitivities Water Score Daily Average', 
                         xlim=[0,365], ylim=[-1,1], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
colors = ['m', 'c', 'r', 'k', 'y', 'b', 'g']

norm_sense_water_plus.plot(title='Plus Sensitivities Water Score', 
                         xlim=[0,365], ylim=[-1,1], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
colors = ['m', 'c', 'r', 'k', 'y', 'b', 'g']

norm_sense_water_plus_daily.plot(title='Plus Sensitivities Water Score Daily Average', 
                         xlim=[0,365], ylim=[-1,1], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
anthro_water = pd.concat([norm_sense_water_min_daily[["Anthropogenic Heat"]],norm_sense_water_plus_daily[["Anthropogenic Heat"]]], axis=1)
anthro_water.columns = ["Anthropogenic Heat Minus", "Anthropogenic Heat Plus"]
anthro_water.plot(title='Anthro Sensitivities Water Score', 
                         xlim=[0,365], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%

#        'Objective Hysteresis Model', 'Sub-surface Soil Character',
#        'Surface Reflectance characteristics',
#        'Surface Water Storage characteristics', 'Vegetation characteristics'],
#       dtype='object')
surf_cond_water = pd.concat([norm_sense_water_min_daily[["Surface Conductance"]],norm_sense_water_plus_daily[["Surface Conductance"]]], axis=1)
surf_cond_water.columns = ["Surface Conductance Minus", "Surface Conductance Plus"]
surf_cond_water.plot(title='Surface Conductance Sensitivities Water Score', 
                         xlim=[0,365], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
#        'Objective Hysteresis Model', 'Sub-surface Soil Character',
#        'Surface Reflectance characteristics',
#        'Surface Water Storage characteristics', 'Vegetation characteristics'],
#       dtype='object')
OHM_water = pd.concat([norm_sense_water_min_daily[["Objective Hysteresis Model"]],norm_sense_water_plus_daily[["Objective Hysteresis Model"]]], axis=1)
OHM_water.columns = ["Objective Hysteresis Model Minus", "Objective Hysteresis Model Plus"]
OHM_water.plot(title='Objective Hysteresis Model Sensitivities Water Score', 
                         xlim=[0,365], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
#        'Surface Reflectance characteristics',
#        'Surface Water Storage characteristics', 'Vegetation characteristics'],
#       dtype='object')
soil_water = pd.concat([norm_sense_water_min_daily[["Sub-surface Soil Character"]],norm_sense_water_plus_daily[["Sub-surface Soil Character"]]], axis=1)
soil_water.columns = ["Sub-surface Soil Character Minus", "Sub-surface Soil Character Plus"]
soil_water.plot(title='Sub-surface Soil Character Sensitivities Water Score', 
                         xlim=[0,365], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
#        'Surface Water Storage characteristics', 'Vegetation characteristics'],
#       dtype='object')
surf_Reflect_water = pd.concat([norm_sense_water_min_daily[["Surface Reflectance characteristics"]],norm_sense_water_plus_daily[["Surface Reflectance characteristics"]]], axis=1)
surf_Reflect_water.columns = ["Surface Reflectance characteristics Minus", "Surface Reflectance characteristics Plus"]
surf_Reflect_water.plot(title='Surface Reflectance characteristics Sensitivities Water Score', 
                         xlim=[0,365], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
surf_storage_water = pd.concat([norm_sense_water_min_daily[["Surface Water Storage characteristics"]],norm_sense_water_plus_daily[["Surface Water Storage characteristics"]]], axis=1)
surf_storage_water.columns = ["Surface Water Storage characteristics Minus", "Surface Water Storage characteristics Plus"]
surf_storage_water.plot(title='Surface Water Storage characteristics Sensitivities Water Score', 
                         xlim=[0,365], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %%
#      'Vegetation characteristics'],
#       dtype='object')
veg_water = pd.concat([norm_sense_water_min_daily[["Vegetation characteristics"]],norm_sense_water_plus_daily[["Vegetation characteristics"]]], axis=1)
veg_water.columns = ["Vegetation characteristics Minus", "Vegetation characteristics Plus"]
veg_water.plot(title='Vegetation characteristics Sensitivities Water Score', 
                         xlim=[0,365], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %% [markdown]
# ##### Bulk Surface Temperatures

# %%
norm_minus_Tsurf_SA_df.plot(title='Minus Sensitivities Bulk Surface Temperatures ($T_{surf}$)', 
                         xlim=[0,365], ylim=[-1,1], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=['y','g'])


# %%
norm_plus_Tsurf_SA_df.plot(title='Plus Sensitivities Bulk Surface Temperatures ($T_{surf}$)', 
                        xlim=[0,365], ylim=[-1,1], xlabel='time (days)', 
                        ylabel='Normalized Sensitivity', color=['y','g'])


# %% [markdown]
# From the above graphs it can be seen that the bulk surface temperatures are directly proportional to the surface reflectance characteristics and inversely proportional to the vegetation characteristics (as long as they are increased, no noticeable difference when they are decreased).

# %% [markdown]
# ##### Storage Heat Flux

# %%

norm_minus_QS_SA_df.plot(title='Minus Sensitivities Storage Heat Flux ($Q_{S}$)', 
                         xlim=[0,365], ylim=[-1,1], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=['r','y','g'])


# %%
norm_plus_QS_SA_df.plot(title='Plus Sensitivities Storage Heat Flux ($Q_{S}$)', 
                        xlim=[0,365], ylim=[-1,1], xlabel='time (days)', 
                        ylabel='Normalized Sensitivity', color=['r','y','g'])

# %% [markdown]
# The storage heat flux is very senstive to any changes in the objective hysteresis model parameters. They react in a somewhat inversely proportional manner, as in if you increase them they will start going more negative in their daily cycles, however on the flip side, they will

# %%
colors = ['m', 'c', 'r', 'k', 'y', 'b', 'g']

norm_minus_QH_SA_df.plot(title='Minus Sensitivities Sensible Heat Flux ($Q_{H}$)', 
                         xlim=[0,365], ylim=[-1,1], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

norm_minus_QH_SA_df[['Anthropogenic Heat','Surface Conductance','Vegetation characteristics']].plot(title='Minus Sensitivities Sensible Heat Flux ($Q_{H}$)', 
                         xlim=[0,365], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=['m', 'c', 'g'])


# %%
norm_plus_QH_SA_df.plot(title='Plus Sensitivities Sensible Heat Flux ($Q_{H}$)', 
                        xlim=[0,365], ylim=[-1,1], xlabel='time (days)', 
                        ylabel='Normalized Sensitivity', color=colors)


# %%

norm_minus_QE_SA_df.plot(title='Minus Sensitivities Latent Heat Flux ($Q_{E}$)', 
                         xlim=[0,365], ylim=[-1,1], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)


# %%
norm_plus_QE_SA_df.plot(title='Plus Sensitivities Latent Heat Flux ($Q_{E}$)', 
                        xlim=[0,365], ylim=[-1,1], xlabel='time (days)', 
                        ylabel='Normalized Sensitivity', color=colors)


# %%

norm_minus_Evap_SA_df.plot(title='Minus Sensitivities Evaporation Rate', 
                           xlim=[0,365], ylim=[-1,1], xlabel='time (days)', 
                           ylabel='Normalized Sensitivity', color=colors)


# %%
norm_plus_Evap_SA_df.plot(title='Plus Sensitivities Evaporation Rate',
                          xlim=[0,365], ylim=[-1,1], xlabel='time (days)',
                          ylabel='Normalized Sensitivity', color=colors)


# %%

norm_minus_RO_SA_df.plot(title='Minus Sensitivities Runoff Rate', xlim=[0,365], 
                         ylim=[-1,1], xlabel='time (days)', 
                         ylabel='Normalized Sensitivity', color=colors)


# %%
norm_plus_RO_SA_df.plot(title='Plus Sensitivities Runoff Rate', xlim=[0,365], 
                        ylim=[-1,1], xlabel='time (days)', 
                        ylabel='Normalized Sensitivity', color=colors)


# %%

norm_minus_TotCh_SA_df.plot(title='Minus Sensitivities Surface and Soil Moisture Change', 
                            xlim=[0,365], ylim=[-1,1], xlabel='time (days)', 
                            ylabel='Normalized Sensitivity', color=colors)


# %%
norm_plus_TotCh_SA_df.plot(title='Plus Sensitivities Surface and Soil Moisture Change',
                           xlim=[0,365], ylim=[-1,1], xlabel='time (days)',
                           ylabel='Normalized Sensitivity', color=colors)


# %%

norm_minus_Drainage_SA_df.plot(title='Minus Sensitivities Drainage Rate', xlim=[0,365], 
                               ylim=[-1,1], xlabel='time (days)', 
                               ylabel='Normalized Sensitivity', color=colors)


# %%
norm_plus_Drainage_SA_df.plot(title='Plus Sensitivities Drainage Rate', xlim=[0,365],
                              ylim=[-1,1], xlabel='time (days)',
                              ylabel='Normalized Sensitivity', color=colors)


# %%

norm_minus_SMD_SA_df.plot(title='Minus Sensitivities Soil Moisture Deficit', 
                          xlim=[0,365], ylim=[-1,1], xlabel='time (days)', 
                          ylabel='Normalized Sensitivity', color=colors)


# %%
norm_plus_SMD_SA_df.plot(title='Plus Sensitivities Soil Moisture Deficit', 
                         xlim=[0,365], ylim=[-1,1], xlabel='time (days)',
                         ylabel='Normalized Sensitivity', color=colors)

# %% [markdown]
# ##### Validation UMEP

# %% [markdown]
# Strategy for showing sensitivities - create a DF per parameter; label based on what it represents, positive and negative; normalize it based on max an min variables; then plot it over time.

# %%
groundTruth_gdf = gpd.read_file(UMEP_ANALYSIS_folder + "heat_severitiy_LA_2023.gpkg")

# %%
groundTruth_gdf.head(1)

# %%
# normGroundTruth = groundTruth_gdf[["UHII","geometry"]].copy()
# normGroundTruth.UHII = (normGroundTruth.UHII - np.min(normGroundTruth.UHII))/(np.max(normGroundTruth.UHII)- np.min(normGroundTruth.UHII))
groundTruth = groundTruth_gdf[["gridcode","geometry"]].copy()
groundTruth["normSeverity"] = (groundTruth.gridcode - np.min(groundTruth.gridcode))/(np.max(groundTruth.gridcode)- np.min(groundTruth.gridcode))
# groundTruth_gdf.head(1)

# %%
envPredScoreGrid = gpd.read_parquet(folder + "envPredGridScored")

# %%
envHeatScoreGrid = envPredScoreGrid[["heat_score","geometry"]].copy()
normHeatScoreGrid = envHeatScoreGrid.copy()
normHeatScoreGrid.heat_score = (normHeatScoreGrid.heat_score - np.min(normHeatScoreGrid.heat_score))/(np.max(normHeatScoreGrid.heat_score) - np.min(normHeatScoreGrid.heat_score))

# %%
groundTruth = groundTruth.to_crs(envHeatScoreGrid.crs)

# %%
envHeatScoreGridv2 = normHeatScoreGrid.overlay(groundTruth, how="intersection")
envHeatScoreGridv2[["normSeverity","heat_score"]].corr(method='kendall')

# %%
heat_score_checking = area_interpolate(normHeatScoreGrid, groundTruth, intensive_variables=["heat_score"], n_jobs=1)
# heat_score_checking_norm = heat_score_checking.copy()

# %%
envHeatScoreCheck = groundTruth[["normSeverity","gridcode","geometry"]].copy()
envHeatScoreCheck = envHeatScoreCheck.to_crs(envHeatScoreGrid.crs)
envHeatScoreCheck = heat_score_checking.merge(envHeatScoreCheck)

# %%
# heat_score_checking_norm.heat_score = (heat_score_checking.heat_score - np.min(heat_score_checking.heat_score))/(np.max(heat_score_checking.heat_score) - np.min(heat_score_checking.heat_score))
normGroundTruth = normGroundTruth.merge(heat_score_checking_norm, on="geometry")
# normGroundTruth.drop(columns=["heat_score_x", "heat_score_y"], inplace=True)
normGroundTruth

# %%
envHeatScoreCheck[["gridcode","heat_score"]].corr(method='kendall')

# %%
enclosureNum = len(bv1.eID.unique())
tessNum = len(bv1)
tessPerEnc = tessNum/enclosureNum
totalBuildingAreaProportion = np.sum(bv1.sicCAR)/tessNum

# %%
print("Tessellations per enclosure: " + str(tessPerEnc))
print("Total area covered by buildings in all tessellations: " + str(totalBuildingAreaProportion*100) + "%")

# %%
extent = gpd.read_file("work/data_for_spatial_sig/LosAngelesExtent/Outline_of_SA.gpkg") # define extent to investigate -> 


# %%
totalCover.values

# %%
tessAreas = bv1.overlay(extent, how="intersection")
AOI_area = extent.area
tessAreas = np.sum(tessAreas.area)
totalCover = tessAreas/AOI_area
print("Proportion of Area covered by Tesselation cells: " + str(totalCover.values[0]*100) + "%")

# %% [markdown]
# # 6. Figures
# 
# Finally, we create final figures.

# %%
final = gpd.read_parquet(folder + "signatures.pq")

# %%
enc = gpd.read_parquet(folder + "enclosures.pq")

# %%
sns.set(context="paper", style="ticks", rc={'patch.force_edgecolor': False})

# %%
cmap = ugg.get_colormap(final.cluster.nunique(), randomize=False)

# %%
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

fig, ax = plt.subplots(figsize=(8, .5))
ax.imshow(gradient, aspect='auto', cmap=cmap)

# %%
cols = cmap.colors
symbology = {0: cols[3],
             1: cols[9],
             2: cols[2],
             3: cols[10],
             4: cols[5],
             5: cols[0],
             6: cols[11],
             7: cols[4],
             8: cols[7],
             9: cols[8],
            }

# %%
df = final.set_crs(enc.crs).to_crs(3857)

# %%
token = ""

# %%
ax = df.plot(color=df['cluster'].map(symbology), figsize=(20, 20), zorder=1, linewidth=.3, edgecolor='w', alpha=1)
contextily.add_basemap(ax, crs=df.crs, source=ugg.get_tiles('roads', token), zorder=2, alpha=.3, zoom=10)
contextily.add_basemap(ax, crs=df.crs, source=ugg.get_tiles('labels', token), zorder=3, alpha=1, zoom=10)
contextily.add_basemap(ax, crs=df.crs, source=ugg.get_tiles('background', token), zorder=-1, alpha=1, zoom=10)
ax.set_axis_off()

scalebar = ScaleBar(dx=1,
                    color=ugg.COLORS[0],
                    location='lower right',
                    height_fraction=0.002,
                    pad=.5,
                    frameon=False,
                    )
ax.add_artist(scalebar)

ugg.north_arrow(plt.gcf(), ax, 0, size=.026, linewidth=1, color=ugg.COLORS[0], loc="upper left", pad=.002, alpha=.9)

from matplotlib.lines import Line2D
custom_points = [Line2D([0], [0], marker="o", linestyle="none", markersize=10, color=color) for color in symbology.values()]
leg_points = ax.legend(custom_points, symbology.keys(), loc='upper right', frameon=True)
ax.add_artist(leg_points)
plt.savefig(folder + "signatures.png", dpi=300, bbox_inches="tight")

# %%
tess = gpd.read_parquet(folder + "tess.pq", columns=['buildings']).set_geometry("buildings")
clusters = pd.read_csv(folder + "FINAL_cluster_labels.csv", index_col=0)
tess['cluster'] = clusters.clusters.values

# %%
ax = tess.to_crs(3857).plot(color=tess['cluster'].map(symbology), figsize=(20, 20), zorder=1, alpha=1)
contextily.add_basemap(ax, crs=df.crs, source=ugg.get_tiles('roads', token), zorder=2, alpha=.3, zoom=10)
contextily.add_basemap(ax, crs=df.crs, source=ugg.get_tiles('labels', token), zorder=3, alpha=1, zoom=10)
contextily.add_basemap(ax, crs=df.crs, source=ugg.get_tiles('background', token), zorder=-1, alpha=1, zoom=10)
ax.set_axis_off()

scalebar = ScaleBar(dx=1,
                    color=ugg.COLORS[0],
                    location='lower right',
                    height_fraction=0.002,
                    pad=.5,
                    frameon=False,
                    )
ax.add_artist(scalebar)

ugg.north_arrow(plt.gcf(), ax, 0, size=.026, linewidth=1, color=ugg.COLORS[0], loc="upper left", pad=.002, alpha=.9)

from matplotlib.lines import Line2D
custom_points = [Line2D([0], [0], marker="o", linestyle="none", markersize=10, color=color) for color in symbology.values()]
leg_points = ax.legend(custom_points, symbology.keys(), loc='upper right', frameon=True)
ax.add_artist(leg_points)
plt.savefig(folder + "signatures_buildings.png", dpi=300, bbox_inches="tight")

# %%
clusters = pd.read_csv(folder + "FINAL_cluster_labels.csv", index_col=0)

# %%
counts = clusters.clusters.value_counts(normalize=True)
color = counts.index.map(symbology)
counts.plot.bar(color=color)
plt.ylabel("ratio")
plt.savefig(folder + "counts.pdf", dpi=300, bbox_inches="tight")

# %% [markdown]
# # 7. Cluster summary
# 
# We can also export summary values of our input characters for each cluster.

# %% [markdown]
# We list characters we want to summarise.

# %%
characters = [
    "sdbAre",
    "sdbPer",
    "sdbCoA",
    "ssbCCo",
    "ssbCor",
    "ssbSqu",
    "ssbERI",
    "ssbElo",
    "ssbCCM",
    "ssbCCD",
    "stbOri",
    "sdcLAL",
    "sdcAre",
    "sscCCo",
    "sscERI",
    "stcOri",
    "sicCAR",
    "stbCeA",
    "mtbAli",
    "mtbNDi",
    "mtcWNe",
    "mdcAre",
    "ltcWRE",
    "ltbIBD",
    "bdShDv",
    "ltcRea",
    "ltcAre",
    "sdsSPW",
    "sdsSWD",
    "sdsSPO",
    "sdsLen",
    "sssLin",
    "ldsMSL",
    "mtdDeg",
    "lcdMes",
    "linP3W",
    "linP4W",
    "linPDE",
    "lcnClo",
    "ldsCDL",
    "xcnSCl",
    "mtdMDi",
    "clsCnt",
    "btwCnt",
    "strCnt",
    "lddNDe",
    "linWID",
    "ldeAre",
    "ldePer",
    "lseCCo",
    "lseERI",
    "lseCWA",
    "lteOri",
    "lteWNB",
    "lieWCe",
    "stbSAl",
    "sdsAre",
    "sisBpM",
    "misCel",
    "mdsAre",
    "lisCel",
    "ldsAre",
    "sddAre",
]

# %%
data = dd.read_parquet(folder + "chunked/chunk_*.pq", columns=characters).compute()

# %%
clusters = pd.read_csv(folder + "FINAL_cluster_labels_v2.csv", index_col=0)

# %% [markdown]
# We get mean of each character for each cluster.

# %%
data["cluster"] = clusters.clusters.values
means = data[characters].groupby(data.cluster).mean().T

# %% [markdown]
# It is better to use full names as the result should be interpretable. Codes are not that useful.

# %%
char_used_names = {
    'sdbCoA': 'courtyard area of building',
    'sscCCo': 'circular compactness of ETC', #
    'stcOri': 'orientation of ETC', #
    'sicCAR': 'covered area ratio of ETC', # building cover
    'stbCeA': 'cell alignment of building',
    'mtbAli': 'alignment of neighbouring buildings',
    'mtbNDi': 'mean distance between neighbouring buildings',
    'mtcWNe': 'perimeter-weighted neighbours of ETC', 
    'ltcWRE': 'weighted reached enclosures of ETC',
    'ltbIBD': 'mean inter-building distance',
    'sdsSPW': 'width of street profile',
    'sdsSWD': 'width deviation of street profile',
    'sdsSPO': 'openness of street profile', #
    'sssLin': 'linearity of street segment', #
    'ldsMSL': 'mean segment length within 3 steps',
    'mtdDeg': 'node degree of junction',
    'lcdMes': 'local meshedness of street network',
    'lcnClo': 'local closeness of street network',
    'xcnSCl': 'square clustering of street network',
    'mtdMDi': 'mean distance to neighbouring nodes of street network',
    'lddNDe': 'local node density of street network',
    'linWID': 'local degree weighted node density of street network',
    'stcSAl': 'street alignment of ETC', 
    'mtbSWR': 'shared walls ratio of buildings',
    'sddAre': 'area covered by node-attached ETCs',
    'sdsAre': 'area covered by edge-attached ETCs',
    'sisBpM': 'buildings per meter of street segment',
    'misCel': 'reached ETCs by neighbouring segments',
    'mdsAre': 'reached area by neighbouring segments',
    'lisCel': 'reached ETCs by local street network',
    'ldsAre': 'reached area by local street network',
    'ltcRea': 'reached ETCs by tessellation contiguity',
    'ldePer': 'perimeter of enclosure',
    'lseCCo': 'circular compactness of enclosure',
    'lseERI': 'equivalent rectangular index of enclosure',
    'lseCWA': 'compactness-weighted axis of enclosure',
    'lteOri': 'orientation of enclosure',
    'lteWNB': 'perimeter-weighted neighbours of enclosure',
    'lieWCe': 'area-weighted ETCs of enclosure', # 
    'ldbPWL': 'perimeter wall length of adjacent buildings',
    'libNCo': 'number of courtyards within adjacent buildings',
    'misRea': 'cells reached within neighbouring street segments'
}

# %%
char_names = {
    'sdbAre': 'area of building',
    'sdbPer': 'perimeter of building',
    'sdbCoA': 'courtyard area of building',
    'ssbCCo': 'circular compactness of building', 
    'ssbCor': 'corners of building',
    'ssbSqu': 'squareness of building',
    'ssbERI': 'equivalent rectangular index of building',
    'ssbElo': 'elongation of building',
    'ssbCCM': 'centroid - corner mean distance of building',
    'ssbCCD': 'centroid - corner distance deviation of building',
    'stbOri': 'orientation of building',
    'sdcLAL': 'longest axis length of ETC',
    'sdcAre': 'area of ETC', # not in score
    'sscCCo': 'circular compactness of ETC', 
    'sscERI': 'equivalent rectangular index of ETC',
    'stcOri': 'orientation of ETC', #
    'sicCAR': 'covered area ratio of ETC', # building cover
    'stbCeA': 'cell alignment of building',
    'mtbAli': 'alignment of neighbouring buildings',
    'mtbNDi': 'mean distance between neighbouring buildings',
    'mtcWNe': 'perimeter-weighted neighbours of ETC', 
    'mdcAre': 'area covered by neighbouring cells',
    'ltcWRE': 'weighted reached enclosures of ETC',
    'ltbIBD': 'mean inter-building distance',
    'sdsSPW': 'width of street profile',
    'sdsSWD': 'width deviation of street profile',
    'sdsSPO': 'openness of street profile', #
    'sdsLen': 'length of street segment',
    'sssLin': 'linearity of street segment', #
    'ldsMSL': 'mean segment length within 3 steps',
    'mtdDeg': 'node degree of junction',
    'lcdMes': 'local meshedness of street network',
    'linP3W': 'local proportion of 3-way intersections of street network',
    'linP4W': 'local proportion of 4-way intersections of street network',
    'linPDE': 'local proportion of cul-de-sacs of street network',
    'lcnClo': 'local closeness of street network',
    'ldsCDL': 'local cul-de-sac length of street network',
    'xcnSCl': 'square clustering of street network',
    'mtdMDi': 'mean distance to neighbouring nodes of street network',
    'clsCnt': 'closeness centrality of nodes in street network', #dropped for ss
    'btwCnt': 'betweenness centrality of nodes in street network',
    'strCnt': 'straightness centrality of nodes in street network',
    'lddNDe': 'local node density of street network',
    'linWID': 'local degree weighted node density of street network',
    'stbSAl': 'street alignment of building',
    'stcSAl': 'street alignment of ETC', 
    'mtbSWR': 'shared walls ratio of buildings',
    'sddAre': 'area covered by node-attached ETCs',
    'sdsAre': 'area covered by edge-attached ETCs',
    'sisBpM': 'buildings per meter of street segment',
    'misCel': 'reached ETCs by neighbouring segments',
    'mdsAre': 'reached area by neighbouring segments',
    'lisCel': 'reached ETCs by local street network',
    'ldsAre': 'reached area by local street network',
    'bdShDv': 'shannon diversity index for buildings',
    'ltcRea': 'reached ETCs by tessellation contiguity',
    'ltcAre': 'reached area by tessellation contiguity',
    'ldeAre': 'area of enclosure',
    'ldePer': 'perimeter of enclosure',
    'lseCCo': 'circular compactness of enclosure',
    'lseERI': 'equivalent rectangular index of enclosure',
    'lseCWA': 'compactness-weighted axis of enclosure',
    'lteOri': 'orientation of enclosure',
    'lteWNB': 'perimeter-weighted neighbours of enclosure',
    'lieWCe': 'area-weighted ETCs of enclosure', # 
    'ldbPWL': 'perimeter wall length of adjacent buildings',
    'libNCo': 'number of courtyards within adjacent buildings',
    'misRea': 'cells reached within neighbouring street segments'
}

# %%
means.index = means.index.map(char_names)

# %% [markdown]
# We do the same with function characters.

# %%
population = pd.read_parquet(folder + "population/population.pq")
workplace = pd.read_parquet(folder + "workplace.pq")
night_lights = pd.read_parquet(folder + "night_lights/night_lights.pq")
historical = dd.read_parquet(folder + "historical/*.pq").compute()
ndvi = dd.read_parquet(folder + "ndvi/*.pq").compute()
historical = historical.set_index('tID').sort_index().fillna(0)
ndvi = ndvi.set_index('tID').sort_index()

# %%
function = workplace.merge(night_lights, on='tID').merge(historical, on='tID').merge(ndvi, on='tID').merge(population, on='tID').set_index('tID')

# %%
function.columns = ['Workplace population - class CNS01', 'Workplace population - class CNS02', 'Workplace population - class CNS03', 'Workplace population - class CNS04', 'Workplace population - class CNS05', 'Workplace population - class CNS06', 'Workplace population - class CNS07', 'Workplace population - class CNS08',
       'Workplace population - class CNS09', 'Workplace population - class CNS10', 'Workplace population - class CNS11', 'Workplace population - class CNS12', 'Workplace population - class CNS13', 'Workplace population - class CNS14', 'Workplace population - class CNS15', 'Workplace population - class CNS16',
       'Workplace population - class CNS17', 'Workplace population - class CNS18', 'Workplace population - class CNS19', 'Workplace population - class CNS20', 'night lights', 'historical sites',
       'NDVI', 'population']

# %%
f_means = function.groupby(data.cluster.values).mean().T

# %% [markdown]
# Finally, we join form and function summaries and plot the final output as a styled pandas DataFrame.

# %%
together = means.append(f_means)

# %%
together.fillna(0).style.format('{:.2f}').background_gradient(cmap=sns.diverging_palette(235, 16, 60, 55, as_cmap=True) ,axis=1,)

# %% [markdown]
# ## Create a data product
# 
# Create files to be uploaded to figshare for public access. We need the raw measured data on ET level and signature geometry.

# %%
data = dask_geopandas.read_parquet(folder + "chunked/chunk_*.pq", columns=characters + ["buildings", "tessellation"]).compute()

# %%
conv = dd.read_parquet(folder + "convolutions/*").compute()

# %%
population = pd.read_parquet(folder + "population/population.pq")
workplace = pd.read_parquet(folder + "workplace.pq")
night_lights = pd.read_parquet(folder + "night_lights/night_lights.pq")
historical = dd.read_parquet(folder + "historical/*.pq").compute()
ndvi = dd.read_parquet(folder + "ndvi/*.pq").compute()
historical = historical.set_index('tID').sort_index().fillna(0)
ndvi = ndvi.set_index('tID').sort_index()

function = workplace.merge(night_lights, on='tID').merge(historical, on='tID').merge(ndvi, on='tID').merge(population, on='tID').set_index('tID')

function.columns = ['Workplace population - class CNS01', 'Workplace population - class CNS02', 'Workplace population - class CNS03', 'Workplace population - class CNS04', 'Workplace population - class CNS05', 'Workplace population - class CNS06', 'Workplace population - class CNS07', 'Workplace population - class CNS08',
       'Workplace population - class CNS09', 'Workplace population - class CNS10', 'Workplace population - class CNS11', 'Workplace population - class CNS12', 'Workplace population - class CNS13', 'Workplace population - class CNS14', 'Workplace population - class CNS15', 'Workplace population - class CNS16',
       'Workplace population - class CNS17', 'Workplace population - class CNS18', 'Workplace population - class CNS19', 'Workplace population - class CNS20', 'night lights', 'historical sites',
       'NDVI', 'population']

# %%
combined = pd.concat([data.reset_index(drop=True), conv.reset_index(drop=True), function.reset_index(drop=True)], axis=1)

# %%
clusters = pd.read_csv(folder + "FINAL_cluster_labels_v2.csv", index_col=0)

# %%
combined["signature_type"] = clusters.clusters.values

# %%
combined.to_parquet(folder + "houston_data.parquet")

# %%
sig = gpd.read_parquet(folder + "signatures.pq")

# %%
sig.crs = combined.crs

# %%
sig = sig.rename(columns={"cluster":"signature_type"})

# %%
sig.to_parquet(folder + "signatures.pq")


