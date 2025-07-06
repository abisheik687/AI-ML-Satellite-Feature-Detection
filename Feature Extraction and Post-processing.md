
# Feature Extraction and Post-processing

## 1. Glacial Lakes: Detection, Delineation, and Temporal Monitoring

### a. Detection and Delineation
*   **Process:** The trained U-Net, DeepLabV3+, or SAM models will be applied to preprocessed optical (Sentinel-2, Resourcesat-2/2A, Landsat) and SAR (Sentinel-1) imagery.
*   **Output:** Initial raster masks indicating glacial lake presence.
*   **Refinement:** Morphological operations (e.g., opening, closing) will be applied to smooth boundaries, remove small spurious detections, and fill small holes within detected lakes.

### b. Temporal Monitoring
*   **Process:** For temporal monitoring, the trained model will be applied to time-series imagery of the same region.
*   **Change Detection:** Pixel-wise or object-based comparison of lake masks across different time steps will identify changes in lake area, formation of new lakes, or disappearance of existing ones.
*   **Quantification:** The area of each detected lake will be calculated for each time step, allowing for quantitative analysis of temporal evolution.
*   **Output:** Temporal change maps showing expansion, contraction, or new lake formation, along with tabular data on lake area over time.

## 2. Road Centrelines: Extraction and Skeletonization

### a. Initial Segmentation
*   **Process:** The trained U-Net, DeepLabV3+, or SAM models will be applied to high-resolution optical imagery (LISS-IV, Sentinel-2) to segment road pixels.
*   **Output:** Binary raster masks where road pixels are identified.

### b. Skeletonization and Centreline Extraction
*   **Process:** Post-processed skeletonization techniques will be applied to the segmented road masks to extract continuous centrelines.
    *   **Thinning Algorithms:** Algorithms like `skimage.morphology.skeletonize` will reduce the segmented road regions to single-pixel-wide lines, representing the centrelines.
    *   **Spur Removal:** Small branches or spurs resulting from the skeletonization process will be removed to ensure smooth and continuous centrelines.
    *   **Gap Filling:** Small gaps in the centrelines will be filled to maintain connectivity, especially in areas with occlusions or image artifacts.
*   **Output:** Raster representation of road centrelines.

## 3. Urban Drainage Systems: Hydrological Modeling and ML Classification

### a. Hydrological Modeling on DEMs
*   **Process:** Conditioned DEMs (SRTM, CartoDEM, ASTER) will be used to derive hydrological features.
    *   **Flow Direction:** Determine the direction of water flow from each cell to its steepest downslope neighbor.
    *   **Flow Accumulation:** Calculate the accumulated upslope area for each cell, indicating potential stream paths.
    *   **Stream Delineation:** Apply a threshold to the flow accumulation raster to delineate initial stream networks.
*   **Tools:** WhiteboxTools or GRASS GIS for these operations.

### b. ML-based Classification for Refinement
*   **Process:** Machine learning models (Random Forest, SVM) will be trained using features derived from satellite imagery (spectral bands, indices like NDWI, texture features) and DEMs (slope, aspect, curvature).
*   **Purpose:** To classify pixels as part of the urban drainage network, refining the hydrological model outputs and accounting for features not captured by DEMs alone (e.g., man-made drainage structures).
*   **Integration:** The results from hydrological modeling and ML classification will be combined to produce a robust and connected urban drainage network.
*   **Output:** Raster representation of urban drainage systems.

## 4. Conversion to Shapefile/GeoJSON Formats

*   **Process:** All extracted features (glacial lake polygons, road centrelines, urban drainage networks) will be converted from raster format to vector formats (Shapefile or GeoJSON) for use in GIS applications.
*   **Tools:** `GDAL/OGR` and `GeoPandas` will be used for this conversion.
    *   **Polygonization:** For glacial lakes and urban drainage (if represented as polygons), raster masks will be converted to polygons.
    *   **Vectorization of Lines:** For road centrelines and urban drainage (if represented as lines), raster lines will be vectorized into line features.
*   **Attributes:** Relevant attributes (e.g., lake area, date of detection, road type, stream order) will be added to the vector features.
*   **Output:** Final Shapefile (.shp, .shx, .dbf, .prj) or GeoJSON (.geojson) files for each feature type, ready for GIS analysis and visualization.

