# Implementation Guide and Code Examples

## Table of Contents
1. [Installation and Setup](#installation-and-setup)
2. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
3. [Model Training Examples](#model-training-examples)
4. [Feature Extraction Implementation](#feature-extraction-implementation)
5. [Validation and Evaluation](#validation-and-evaluation)
6. [Dashboard Deployment](#dashboard-deployment)
7. [API Usage Examples](#api-usage-examples)

## Installation and Setup

### System Requirements
```bash
# Minimum system requirements
# - Ubuntu 20.04 or later
# - Python 3.8+
# - CUDA 11.8+ (for GPU support)
# - 16GB RAM (32GB recommended)
# - 100GB free disk space

# Check system compatibility
python3 --version
nvidia-smi  # Check GPU availability
```

### Environment Setup
```bash
# Create virtual environment
python3 -m venv geospatial_env
source geospatial_env/bin/activate

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow-gpu==2.13.0
pip install rasterio gdal geopandas shapely fiona
pip install scikit-image opencv-python-headless
pip install plotly streamlit folium
pip install whitebox whiteboxgui
```

### Package Installation
```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.13.0
rasterio>=1.3.0
gdal>=3.4.0
geopandas>=0.13.0
shapely>=2.0.0
fiona>=1.9.0
scikit-image>=0.20.0
opencv-python-headless>=4.7.0
plotly>=5.14.0
streamlit>=1.22.0
folium>=0.14.0
whitebox>=2.2.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Data Preprocessing Pipeline

### Satellite Data Acquisition
```python
import ee
import rasterio
import numpy as np
from datetime import datetime, timedelta

class SatelliteDataAcquisition:
    def __init__(self, service_account_key=None):
        """Initialize Google Earth Engine connection."""
        if service_account_key:
            credentials = ee.ServiceAccountCredentials(
                email='your-service-account@project.iam.gserviceaccount.com',
                key_file=service_account_key
            )
            ee.Initialize(credentials)
        else:
            ee.Initialize()
    
    def get_sentinel2_collection(self, aoi, start_date, end_date, cloud_cover=20):
        """Acquire Sentinel-2 imagery for specified area and time range."""
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(aoi)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
                     .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'QA60']))
        
        return collection
    
    def get_sentinel1_collection(self, aoi, start_date, end_date):
        """Acquire Sentinel-1 SAR imagery."""
        collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                     .filterBounds(aoi)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                     .filter(ee.Filter.eq('instrumentMode', 'IW'))
                     .select(['VV', 'VH']))
        
        return collection
    
    def download_image(self, image, region, scale=10, filename=None):
        """Download image to local storage."""
        url = image.getDownloadURL({
            'region': region,
            'scale': scale,
            'format': 'GeoTIFF'
        })
        
        # Download implementation would go here
        return url

# Usage example
aoi = ee.Geometry.Rectangle([83.5, 27.8, 84.5, 28.8])  # Nepal region
start_date = '2023-01-01'
end_date = '2023-12-31'

data_acquisition = SatelliteDataAcquisition()
s2_collection = data_acquisition.get_sentinel2_collection(aoi, start_date, end_date)
s1_collection = data_acquisition.get_sentinel1_collection(aoi, start_date, end_date)
```

### Preprocessing Pipeline
```python
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
import numpy as np
from skimage import morphology
import cv2

class ImagePreprocessor:
    def __init__(self, target_crs='EPSG:4326', target_resolution=10):
        self.target_crs = target_crs
        self.target_resolution = target_resolution
    
    def harmonize_resolution(self, input_path, output_path):
        """Resample image to target resolution."""
        with rasterio.open(input_path) as src:
            # Calculate new dimensions
            new_width = int(src.width * src.res[0] / self.target_resolution)
            new_height = int(src.height * src.res[1] / self.target_resolution)
            
            # Create new transform
            new_transform = rasterio.transform.from_bounds(
                *src.bounds, new_width, new_height
            )
            
            # Resample data
            data = src.read()
            resampled_data = np.zeros((src.count, new_height, new_width), dtype=data.dtype)
            
            reproject(
                data, resampled_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=self.target_crs,
                resampling=Resampling.cubic_spline
            )
            
            # Write output
            with rasterio.open(
                output_path, 'w',
                driver='GTiff',
                height=new_height,
                width=new_width,
                count=src.count,
                dtype=data.dtype,
                crs=self.target_crs,
                transform=new_transform
            ) as dst:
                dst.write(resampled_data)
    
    def cloud_shadow_mask(self, image_path, qa_band_path):
        """Generate cloud and shadow mask from QA band."""
        with rasterio.open(qa_band_path) as qa_src:
            qa_data = qa_src.read(1)
        
        # Sentinel-2 QA60 band bit flags
        cloud_bit = 10
        cirrus_bit = 11
        
        # Create cloud mask
        cloud_mask = (qa_data & (1 << cloud_bit)) != 0
        cirrus_mask = (qa_data & (1 << cirrus_bit)) != 0
        
        # Combine masks
        combined_mask = cloud_mask | cirrus_mask
        
        # Morphological operations to expand mask
        kernel = morphology.disk(3)
        expanded_mask = morphology.binary_dilation(combined_mask, kernel)
        
        return expanded_mask.astype(np.uint8)
    
    def coregister_images(self, reference_path, target_path, output_path):
        """Co-register target image to reference image."""
        with rasterio.open(reference_path) as ref_src:
            ref_data = ref_src.read(1)
            ref_transform = ref_src.transform
            ref_crs = ref_src.crs
        
        with rasterio.open(target_path) as tgt_src:
            tgt_data = tgt_src.read()
            
            # Reproject target to reference grid
            coregistered_data = np.zeros_like(ref_data)
            
            reproject(
                tgt_data, coregistered_data,
                src_transform=tgt_src.transform,
                src_crs=tgt_src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.cubic_spline
            )
            
            # Write output
            with rasterio.open(
                output_path, 'w',
                driver='GTiff',
                height=ref_data.shape[0],
                width=ref_data.shape[1],
                count=1,
                dtype=coregistered_data.dtype,
                crs=ref_crs,
                transform=ref_transform
            ) as dst:
                dst.write(coregistered_data, 1)

# Usage example
preprocessor = ImagePreprocessor()
preprocessor.harmonize_resolution('input_image.tif', 'harmonized_image.tif')
cloud_mask = preprocessor.cloud_shadow_mask('image.tif', 'qa_band.tif')
preprocessor.coregister_images('reference.tif', 'target.tif', 'coregistered.tif')
```

### DEM Conditioning
```python
import whitebox
import numpy as np
import rasterio

class DEMProcessor:
    def __init__(self):
        self.wbt = whitebox.WhiteboxTools()
        self.wbt.verbose = False
    
    def condition_dem(self, input_dem, output_dem):
        """Apply DEM conditioning for hydrological analysis."""
        # Fill depressions
        filled_dem = input_dem.replace('.tif', '_filled.tif')
        self.wbt.fill_depressions(input_dem, filled_dem)
        
        # Breach depressions
        breached_dem = input_dem.replace('.tif', '_breached.tif')
        self.wbt.breach_depressions(filled_dem, breached_dem)
        
        # Smooth DEM
        self.wbt.gaussian_filter(breached_dem, output_dem, sigma=1.0)
        
        return output_dem
    
    def calculate_flow_direction(self, dem_path, output_path):
        """Calculate D8 flow direction."""
        self.wbt.d8_pointer(dem_path, output_path)
        return output_path
    
    def calculate_flow_accumulation(self, flow_dir_path, output_path):
        """Calculate flow accumulation."""
        self.wbt.d8_flow_accumulation(flow_dir_path, output_path)
        return output_path
    
    def extract_streams(self, flow_acc_path, output_path, threshold=1000):
        """Extract stream network from flow accumulation."""
        self.wbt.extract_streams(flow_acc_path, output_path, threshold)
        return output_path

# Usage example
dem_processor = DEMProcessor()
conditioned_dem = dem_processor.condition_dem('raw_dem.tif', 'conditioned_dem.tif')
flow_dir = dem_processor.calculate_flow_direction(conditioned_dem, 'flow_direction.tif')
flow_acc = dem_processor.calculate_flow_accumulation(flow_dir, 'flow_accumulation.tif')
streams = dem_processor.extract_streams(flow_acc, 'streams.tif', threshold=1000)
```

## Model Training Examples

### U-Net Implementation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
```

### Training Pipeline
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split

class GeospatialDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read().astype(np.float32)
        
        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1).astype(np.float32)
        
        # Normalize image
        image = (image - image.mean()) / image.std()
        
        # Convert to tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class ModelTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = DiceLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, epochs=100):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Usage example
image_paths = ['image1.tif', 'image2.tif', ...]  # List of image paths
mask_paths = ['mask1.tif', 'mask2.tif', ...]    # List of mask paths

# Split data
train_images, val_images, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

# Create datasets
train_dataset = GeospatialDataset(train_images, train_masks)
val_dataset = GeospatialDataset(val_images, val_masks)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Initialize model and trainer
model = UNet(n_channels=6, n_classes=1)  # 6 channels for multispectral
trainer = ModelTrainer(model)

# Train model
trainer.train(train_loader, val_loader, epochs=100)
```

## Feature Extraction Implementation

### Glacial Lake Detection
```python
import torch
import rasterio
import numpy as np
from skimage import morphology, measure
import geopandas as gpd
from shapely.geometry import Polygon

class GlacialLakeDetector:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = UNet(n_channels=6, n_classes=1)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()
    
    def preprocess_image(self, image_path):
        """Preprocess satellite image for model input."""
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)
            transform = src.transform
            crs = src.crs
        
        # Normalize
        image = (image - image.mean(axis=(1, 2), keepdims=True)) / image.std(axis=(1, 2), keepdims=True)
        
        return image, transform, crs
    
    def predict_lakes(self, image_path, output_path=None):
        """Detect glacial lakes in satellite image."""
        image, transform, crs = self.preprocess_image(image_path)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(image_tensor)
            prediction = torch.sigmoid(prediction).cpu().numpy()[0, 0]
        
        # Post-process
        binary_mask = prediction > 0.5
        
        # Morphological operations
        kernel = morphology.disk(2)
        cleaned_mask = morphology.binary_opening(binary_mask, kernel)
        cleaned_mask = morphology.binary_closing(cleaned_mask, kernel)
        
        # Remove small objects
        cleaned_mask = morphology.remove_small_objects(cleaned_mask, min_size=100)
        
        if output_path:
            # Save prediction
            with rasterio.open(
                output_path, 'w',
                driver='GTiff',
                height=prediction.shape[0],
                width=prediction.shape[1],
                count=1,
                dtype=np.uint8,
                crs=crs,
                transform=transform
            ) as dst:
                dst.write((cleaned_mask * 255).astype(np.uint8), 1)
        
        return cleaned_mask, transform, crs
    
    def vectorize_lakes(self, mask, transform, crs, min_area=1000):
        """Convert raster mask to vector polygons."""
        # Find contours
        contours = measure.find_contours(mask.astype(np.uint8), 0.5)
        
        polygons = []
        for contour in contours:
            # Convert pixel coordinates to geographic coordinates
            coords = []
            for point in contour:
                x, y = rasterio.transform.xy(transform, point[0], point[1])
                coords.append((x, y))
            
            if len(coords) >= 3:  # Valid polygon needs at least 3 points
                polygon = Polygon(coords)
                if polygon.area > min_area:  # Filter by minimum area
                    polygons.append(polygon)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=crs)
        gdf['area'] = gdf.geometry.area
        gdf['perimeter'] = gdf.geometry.length
        
        return gdf

# Usage example
detector = GlacialLakeDetector('glacial_lake_model.pth')
mask, transform, crs = detector.predict_lakes('satellite_image.tif', 'lake_mask.tif')
lake_polygons = detector.vectorize_lakes(mask, transform, crs)
lake_polygons.to_file('glacial_lakes.shp')
```

### Road Centreline Extraction
```python
from skimage import morphology
import cv2
import networkx as nx

class RoadCenterlineExtractor:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = UNet(n_channels=3, n_classes=1)  # RGB input
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()
    
    def extract_roads(self, image_path):
        """Extract road mask from satellite image."""
        image, transform, crs = self.preprocess_image(image_path)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(image_tensor)
            prediction = torch.sigmoid(prediction).cpu().numpy()[0, 0]
        
        # Threshold and clean
        road_mask = prediction > 0.5
        
        # Morphological cleaning
        kernel = morphology.disk(1)
        road_mask = morphology.binary_opening(road_mask, kernel)
        road_mask = morphology.binary_closing(road_mask, kernel)
        
        return road_mask, transform, crs
    
    def skeletonize_roads(self, road_mask):
        """Extract centerlines from road mask."""
        # Skeletonize
        skeleton = morphology.skeletonize(road_mask)
        
        # Remove spurs
        skeleton = self.remove_spurs(skeleton, spur_length=10)
        
        return skeleton
    
    def remove_spurs(self, skeleton, spur_length=10):
        """Remove short spurs from skeleton."""
        # Convert to graph
        graph = self.skeleton_to_graph(skeleton)
        
        # Find endpoints
        endpoints = [node for node, degree in graph.degree() if degree == 1]
        
        # Remove short paths from endpoints
        for endpoint in endpoints:
            path_length = 0
            current = endpoint
            visited = set()
            
            while current not in visited and graph.degree(current) <= 2:
                visited.add(current)
                neighbors = list(graph.neighbors(current))
                neighbors = [n for n in neighbors if n not in visited]
                
                if not neighbors:
                    break
                
                current = neighbors[0]
                path_length += 1
                
                if path_length >= spur_length:
                    break
            
            if path_length < spur_length:
                graph.remove_nodes_from(visited)
        
        # Convert back to skeleton
        cleaned_skeleton = self.graph_to_skeleton(graph, skeleton.shape)
        
        return cleaned_skeleton
    
    def skeleton_to_graph(self, skeleton):
        """Convert skeleton to NetworkX graph."""
        graph = nx.Graph()
        
        # Find skeleton pixels
        y_coords, x_coords = np.where(skeleton)
        
        # Add nodes
        for y, x in zip(y_coords, x_coords):
            graph.add_node((y, x))
        
        # Add edges (8-connectivity)
        for y, x in zip(y_coords, x_coords):
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    
                    ny, nx = y + dy, x + dx
                    if (ny, nx) in graph.nodes:
                        graph.add_edge((y, x), (ny, nx))
        
        return graph
    
    def graph_to_skeleton(self, graph, shape):
        """Convert NetworkX graph back to skeleton."""
        skeleton = np.zeros(shape, dtype=bool)
        
        for node in graph.nodes:
            y, x = node
            skeleton[y, x] = True
        
        return skeleton
    
    def vectorize_centerlines(self, skeleton, transform, crs):
        """Convert skeleton to vector lines."""
        # Convert to graph
        graph = self.skeleton_to_graph(skeleton)
        
        # Find connected components
        components = list(nx.connected_components(graph))
        
        lines = []
        for component in components:
            if len(component) < 2:
                continue
            
            # Find longest path in component
            subgraph = graph.subgraph(component)
            
            # Simple approach: find path between endpoints
            endpoints = [node for node in component if graph.degree(node) == 1]
            
            if len(endpoints) >= 2:
                try:
                    path = nx.shortest_path(subgraph, endpoints[0], endpoints[1])
                    
                    # Convert to geographic coordinates
                    coords = []
                    for y, x in path:
                        geo_x, geo_y = rasterio.transform.xy(transform, y, x)
                        coords.append((geo_x, geo_y))
                    
                    if len(coords) >= 2:
                        lines.append(coords)
                except nx.NetworkXNoPath:
                    continue
        
        # Create GeoDataFrame
        from shapely.geometry import LineString
        line_geoms = [LineString(coords) for coords in lines if len(coords) >= 2]
        gdf = gpd.GeoDataFrame({'geometry': line_geoms}, crs=crs)
        gdf['length'] = gdf.geometry.length
        
        return gdf

# Usage example
extractor = RoadCenterlineExtractor('road_model.pth')
road_mask, transform, crs = extractor.extract_roads('satellite_image.tif')
skeleton = extractor.skeletonize_roads(road_mask)
centerlines = extractor.vectorize_centerlines(skeleton, transform, crs)
centerlines.to_file('road_centerlines.shp')
```

## Validation and Evaluation

### Validation Framework
```python
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

class ValidationFramework:
    def __init__(self):
        pass
    
    def calculate_iou(self, pred_mask, true_mask):
        """Calculate Intersection over Union."""
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def calculate_pixel_metrics(self, pred_mask, true_mask):
        """Calculate pixel-wise classification metrics."""
        pred_flat = pred_mask.flatten()
        true_flat = true_mask.flatten()
        
        precision = precision_score(true_flat, pred_flat, zero_division=0)
        recall = recall_score(true_flat, pred_flat, zero_division=0)
        f1 = f1_score(true_flat, pred_flat, zero_division=0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'iou': self.calculate_iou(pred_mask, true_mask)
        }
    
    def calculate_buffer_overlap(self, pred_lines, true_lines, buffer_distance=10):
        """Calculate buffer overlap accuracy for line features."""
        if len(pred_lines) == 0 or len(true_lines) == 0:
            return 0.0
        
        # Create buffers
        pred_buffer = unary_union([line.buffer(buffer_distance) for line in pred_lines])
        true_buffer = unary_union([line.buffer(buffer_distance) for line in true_lines])
        
        # Calculate overlap
        intersection = pred_buffer.intersection(true_buffer).area
        pred_area = pred_buffer.area
        
        if pred_area == 0:
            return 0.0
        
        return intersection / pred_area
    
    def calculate_completeness(self, pred_features, true_features, distance_threshold=50):
        """Calculate completeness metric."""
        if len(true_features) == 0:
            return 1.0
        
        detected = 0
        for true_feature in true_features:
            # Check if any predicted feature is within threshold distance
            for pred_feature in pred_features:
                if true_feature.distance(pred_feature) < distance_threshold:
                    detected += 1
                    break
        
        return detected / len(true_features)
    
    def calculate_correctness(self, pred_features, true_features, distance_threshold=50):
        """Calculate correctness metric."""
        if len(pred_features) == 0:
            return 1.0
        
        correct = 0
        for pred_feature in pred_features:
            # Check if any true feature is within threshold distance
            for true_feature in true_features:
                if pred_feature.distance(true_feature) < distance_threshold:
                    correct += 1
                    break
        
        return correct / len(pred_features)

# Usage example
validator = ValidationFramework()

# Load predicted and ground truth data
pred_lakes = gpd.read_file('predicted_lakes.shp')
true_lakes = gpd.read_file('ground_truth_lakes.shp')

# Calculate metrics
completeness = validator.calculate_completeness(
    pred_lakes.geometry, true_lakes.geometry
)
correctness = validator.calculate_correctness(
    pred_lakes.geometry, true_lakes.geometry
)

print(f"Completeness: {completeness:.3f}")
print(f"Correctness: {correctness:.3f}")
```

## Dashboard Deployment

### Streamlit Dashboard
```python
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import pandas as pd

class GeospatialDashboard:
    def __init__(self):
        st.set_page_config(
            page_title="Geospatial Analysis Pipeline",
            page_icon="🛰️",
            layout="wide"
        )
    
    def main(self):
        st.title("🛰️ AI/ML-Powered Geospatial Analysis Pipeline")
        
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Select Page",
            ["Overview", "Glacial Lakes", "Road Networks", "Drainage Systems", "Performance"]
        )
        
        if page == "Overview":
            self.overview_page()
        elif page == "Glacial Lakes":
            self.glacial_lakes_page()
        elif page == "Road Networks":
            self.road_networks_page()
        elif page == "Drainage Systems":
            self.drainage_systems_page()
        elif page == "Performance":
            self.performance_page()
    
    def overview_page(self):
        st.header("System Overview")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Glacial Lakes Detected", "1,247", "+12%")
        
        with col2:
            st.metric("Road Network Coverage", "89.3%", "+2.1%")
        
        with col3:
            st.metric("Drainage Systems Mapped", "456", "0%")
        
        with col4:
            st.metric("Processing Speed", "2.3s", "-15%")
        
        # Performance chart
        st.subheader("Model Performance Overview")
        
        performance_data = pd.DataFrame({
            'Feature Type': ['Glacial Lakes', 'Road Centrelines', 'Urban Drainage'],
            'Accuracy (%)': [87, 89, 86],
            'IoU Score': [0.87, 0.85, 0.82]
        })
        
        fig = px.bar(
            performance_data,
            x='Feature Type',
            y='Accuracy (%)',
            title='Model Accuracy by Feature Type'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def glacial_lakes_page(self):
        st.header("Glacial Lakes Detection & Monitoring")
        
        # Time period selector
        col1, col2 = st.columns([3, 1])
        
        with col2:
            year = st.slider("Select Year", 2020, 2024, 2024)
        
        with col1:
            # Create map
            m = folium.Map(location=[28.2380, 83.9956], zoom_start=8)
            
            # Add sample glacial lakes
            lakes = [
                {"lat": 28.2380, "lng": 83.9956, "name": "Lake A", "area": "2.3 km²"},
                {"lat": 28.3000, "lng": 84.1000, "name": "Lake B", "area": "1.8 km²"},
                {"lat": 28.1500, "lng": 83.8000, "name": "Lake C", "area": "3.1 km²"}
            ]
            
            for lake in lakes:
                folium.CircleMarker(
                    location=[lake["lat"], lake["lng"]],
                    radius=10,
                    popup=f"{lake['name']}<br>Area: {lake['area']}",
                    color='blue',
                    fillColor='blue',
                    fillOpacity=0.6
                ).add_to(m)
            
            st_folium(m, width=700, height=500)
        
        # Temporal analysis
        st.subheader("Temporal Analysis")
        
        years = [2020, 2021, 2022, 2023, 2024]
        lake_count = [1156, 1189, 1203, 1231, 1247]
        total_area = [45.2, 46.8, 47.1, 48.3, 49.1]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, y=lake_count,
            mode='lines+markers',
            name='Lake Count',
            yaxis='y'
        ))
        fig.add_trace(go.Scatter(
            x=years, y=total_area,
            mode='lines+markers',
            name='Total Area (km²)',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Glacial Lake Evolution Over Time',
            xaxis_title='Year',
            yaxis=dict(title='Lake Count', side='left'),
            yaxis2=dict(title='Total Area (km²)', side='right', overlaying='y')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def performance_page(self):
        st.header("Pipeline Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Computational Efficiency")
            
            stages = ['Preprocessing', 'Model Inference', 'Post-processing', 'Vectorization']
            times = [0.8, 2.3, 0.5, 0.3]
            
            fig = px.bar(
                x=stages, y=times,
                title='Processing Time by Stage',
                labels={'x': 'Pipeline Stage', 'y': 'Time (seconds)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Model Accuracy Comparison")
            
            models = ['U-Net', 'DeepLabV3+', 'SAM', 'Random Forest']
            accuracy = [85, 89, 87, 78]
            
            fig = px.bar(
                x=models, y=accuracy,
                title='Model Performance Comparison',
                labels={'x': 'Model Architecture', 'y': 'Accuracy (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)

# Run dashboard
if __name__ == "__main__":
    dashboard = GeospatialDashboard()
    dashboard.main()
```

### Flask API Backend
```python
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import geopandas as gpd
import rasterio

app = Flask(__name__)
CORS(app)

class GeospatialAPI:
    def __init__(self):
        self.glacial_detector = GlacialLakeDetector('models/glacial_lake_model.pth')
        self.road_extractor = RoadCenterlineExtractor('models/road_model.pth')
    
    def process_glacial_lakes(self, image_path):
        """Process glacial lake detection."""
        mask, transform, crs = self.glacial_detector.predict_lakes(image_path)
        lakes_gdf = self.glacial_detector.vectorize_lakes(mask, transform, crs)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.geojson', delete=False)
        lakes_gdf.to_file(temp_file.name, driver='GeoJSON')
        
        return temp_file.name, len(lakes_gdf)
    
    def process_roads(self, image_path):
        """Process road centerline extraction."""
        road_mask, transform, crs = self.road_extractor.extract_roads(image_path)
        skeleton = self.road_extractor.skeletonize_roads(road_mask)
        roads_gdf = self.road_extractor.vectorize_centerlines(skeleton, transform, crs)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.geojson', delete=False)
        roads_gdf.to_file(temp_file.name, driver='GeoJSON')
        
        return temp_file.name, len(roads_gdf)

api = GeospatialAPI()

@app.route('/api/process/glacial-lakes', methods=['POST'])
def process_glacial_lakes():
    """API endpoint for glacial lake detection."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    # Save uploaded file
    temp_input = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
    file.save(temp_input.name)
    
    try:
        # Process image
        result_path, lake_count = api.process_glacial_lakes(temp_input.name)
        
        return jsonify({
            'status': 'success',
            'lake_count': lake_count,
            'download_url': f'/api/download/{os.path.basename(result_path)}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        os.unlink(temp_input.name)

@app.route('/api/process/roads', methods=['POST'])
def process_roads():
    """API endpoint for road centerline extraction."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    # Save uploaded file
    temp_input = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
    file.save(temp_input.name)
    
    try:
        # Process image
        result_path, road_count = api.process_roads(temp_input.name)
        
        return jsonify({
            'status': 'success',
            'road_count': road_count,
            'download_url': f'/api/download/{os.path.basename(result_path)}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        os.unlink(temp_input.name)

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download processed results."""
    file_path = os.path.join(tempfile.gettempdir(), filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/status')
def status():
    """API status endpoint."""
    return jsonify({
        'status': 'online',
        'version': '1.0.0',
        'available_endpoints': [
            '/api/process/glacial-lakes',
            '/api/process/roads',
            '/api/download/<filename>',
            '/api/status'
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

## API Usage Examples

### Python Client
```python
import requests
import json

class GeospatialClient:
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
    
    def process_glacial_lakes(self, image_path):
        """Submit glacial lake detection job."""
        url = f"{self.base_url}/api/process/glacial-lakes"
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API error: {response.text}")
    
    def process_roads(self, image_path):
        """Submit road extraction job."""
        url = f"{self.base_url}/api/process/roads"
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API error: {response.text}")
    
    def download_results(self, download_url, output_path):
        """Download processing results."""
        url = f"{self.base_url}{download_url}"
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            return False

# Usage example
client = GeospatialClient()

# Process glacial lakes
result = client.process_glacial_lakes('satellite_image.tif')
print(f"Detected {result['lake_count']} glacial lakes")

# Download results
client.download_results(result['download_url'], 'glacial_lakes_result.geojson')
```

### JavaScript Client
```javascript
class GeospatialClient {
    constructor(baseUrl = 'http://localhost:5000') {
        this.baseUrl = baseUrl;
    }
    
    async processGlacialLakes(imageFile) {
        const formData = new FormData();
        formData.append('image', imageFile);
        
        const response = await fetch(`${this.baseUrl}/api/process/glacial-lakes`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    async processRoads(imageFile) {
        const formData = new FormData();
        formData.append('image', imageFile);
        
        const response = await fetch(`${this.baseUrl}/api/process/roads`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    async downloadResults(downloadUrl) {
        const response = await fetch(`${this.baseUrl}${downloadUrl}`);
        
        if (!response.ok) {
            throw new Error(`Download error: ${response.statusText}`);
        }
        
        return await response.blob();
    }
}

// Usage example
const client = new GeospatialClient();

document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('image-file');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select an image file');
        return;
    }
    
    try {
        const result = await client.processGlacialLakes(file);
        console.log(`Detected ${result.lake_count} glacial lakes`);
        
        // Download results
        const blob = await client.downloadResults(result.download_url);
        const url = URL.createObjectURL(blob);
        
        // Create download link
        const a = document.createElement('a');
        a.href = url;
        a.download = 'glacial_lakes_result.geojson';
        a.click();
        
    } catch (error) {
        console.error('Error:', error);
        alert('Processing failed: ' + error.message);
    }
});
```

This implementation guide provides comprehensive code examples for all major components of the geospatial analysis pipeline, from data preprocessing and model training to feature extraction and deployment. The modular design enables easy customization and extension for specific use cases and geographic regions.

