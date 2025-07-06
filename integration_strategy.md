
# Integration Strategy: Web-Based Geospatial Analysis Pipeline

This document outlines the strategy for integrating the existing HTML/CSS/JavaScript frontend with a new Flask API backend to create a fully web-based geospatial analysis application.

## 1. Backend (Flask API)

### 1.1. Project Structure
- Create a new directory `backend/` at the root of the project.
- The main Flask application will reside in `backend/app.py`.
- A `backend/requirements.txt` file will list all Python dependencies.
- Core geospatial processing logic (e.g., `GlacialLakeDetector`, `RoadCenterlineExtractor`, `DEMProcessor`) will be refactored into separate modules within `backend/src/` for better organization and maintainability.

### 1.2. API Endpoints
- Implement RESTful API endpoints using Flask to expose the geospatial analysis functionalities.
- **`/api/process/glacial-lakes` (POST):** Accepts satellite imagery (e.g., GeoTIFF) as input, processes it for glacial lake detection, and returns the results (e.g., GeoJSON of lake polygons).
- **`/api/process/roads` (POST):** Accepts satellite imagery, extracts road centrelines, and returns GeoJSON of road networks.
- **`/api/process/drainage` (POST):** Accepts satellite imagery and DEM data, delineates urban drainage systems, and returns GeoJSON of stream networks.
- **`/api/download/<filename>` (GET):** Allows downloading of processed results (e.g., GeoJSON files).
- **`/api/status` (GET):** Provides a health check and status of the API.

### 1.3. Data Handling
- The API will handle file uploads (e.g., satellite images) from the frontend.
- Processed results will be temporarily stored on the server and then served as downloadable GeoJSON or Shapefile formats.
- Implement proper error handling and response formatting.

### 1.4. Cross-Origin Resource Sharing (CORS)
- Enable CORS for the Flask application to allow requests from the frontend, which will likely be served from a different origin during development and potentially in production.

## 2. Frontend (HTML/CSS/JavaScript)

### 2.1. Project Structure
- The existing `geospatial-dashboard/` directory will continue to house the frontend assets (`index.html`, `styles.css`, `script.js`).

### 2.2. API Integration
- Modify `geospatial-dashboard/script.js` to:
    - Implement file input elements to allow users to upload satellite imagery.
    - Use the `fetch` API or `XMLHttpRequest` to send uploaded image data to the respective Flask API endpoints.
    - Handle asynchronous responses from the backend.
    - Parse the GeoJSON responses received from the API.
    - Dynamically update the Leaflet maps to visualize the extracted glacial lakes, road centrelines, and urban drainage systems.
    - Display processing status, progress, and any error messages to the user.

### 2.3. User Interface Enhancements
- Add UI elements for file upload (e.g., input type="file", upload button).
- Implement loading indicators while processing is underway.
- Provide options for users to select the type of analysis to perform (glacial lakes, roads, drainage).
- Enhance map interactions to display GeoJSON data effectively.

## 3. Overall Workflow

1.  User uploads satellite imagery via the web dashboard.
2.  Frontend sends the image to the Flask API using an HTTP POST request.
3.  Flask API receives the image, calls the appropriate geospatial processing module (e.g., GlacialLakeDetector).
4.  The processing module performs the analysis and generates GeoJSON results.
5.  Flask API sends the GeoJSON results back to the frontend.
6.  Frontend receives the GeoJSON and renders it on the interactive Leaflet map.
7.  Users can then download the GeoJSON results or view performance metrics.

This strategy ensures a clear separation of concerns between the frontend and backend, facilitating modular development and easier maintenance.

