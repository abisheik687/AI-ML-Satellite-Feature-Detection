from flask import Blueprint, jsonify, request, send_file
from werkzeug.utils import secure_filename
import os
import tempfile
import uuid
import json
import numpy as np
from datetime import datetime

geospatial_bp = Blueprint('geospatial', __name__)

# Configuration
ALLOWED_EXTENSIONS = {'tif', 'tiff', 'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# (This is a placeholder for the actual AI/ML model)
def process_glacial_lakes(image_path):
    """Processes the image to detect glacial lakes."""
    # Simulate a delay
    import time
    time.sleep(np.random.uniform(1.5, 2.5))

    # Simulate a realistic number of lakes
    num_lakes = np.random.randint(5, 15)

    # Generate random lake features
    features = []
    for i in range(num_lakes):
        # Random coordinates
        lon = np.random.uniform(83, 85)
        lat = np.random.uniform(28, 29)

        # Random properties
        area = np.random.uniform(0.5, 5.0)
        perimeter = area * np.random.uniform(2, 4)
        confidence = np.random.uniform(0.8, 0.98)

        # Create a GeoJSON feature
        feature = {
            "type": "Feature",
            "properties": {
                "id": i + 1,
                "area": round(area, 2),
                "perimeter": round(perimeter, 2),
                "confidence": round(confidence, 2)
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon, lat],
                    [lon + 0.001, lat],
                    [lon + 0.001, lat + 0.001],
                    [lon, lat + 0.001],
                    [lon, lat]
                ]]
            }
        }
        features.append(feature)

    # Create the GeoJSON FeatureCollection
    result = {
        "type": "FeatureCollection",
        "features": features
    }

    return result, num_lakes


def process_roads(image_path):
    """Processes the image to extract road centrelines."""
    # Simulate a delay
    import time
    time.sleep(np.random.uniform(1.5, 2.5))

    # Simulate a realistic number of roads
    num_roads = np.random.randint(10, 20)

    # Generate random road features
    features = []
    for i in range(num_roads):
        # Random coordinates
        start_lon = np.random.uniform(77, 78)
        start_lat = np.random.uniform(28, 29)
        end_lon = start_lon + np.random.uniform(-0.1, 0.1)
        end_lat = start_lat + np.random.uniform(-0.1, 0.1)

        # Random properties
        length = np.sqrt((end_lon - start_lon)**2 + (end_lat - start_lat)**2) * 111
        road_type = np.random.choice(['primary', 'secondary', 'tertiary'])
        confidence = np.random.uniform(0.75, 0.95)

        # Create a GeoJSON feature
        feature = {
            "type": "Feature",
            "properties": {
                "id": i + 1,
                "length": round(length, 2),
                "road_type": road_type,
                "confidence": round(confidence, 2)
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [start_lon, start_lat],
                    [end_lon, end_lat]
                ]
            }
        }
        features.append(feature)

    # Create the GeoJSON FeatureCollection
    result = {
        "type": "FeatureCollection",
        "features": features
    }

    return result, num_roads


def process_drainage(image_path):
    """Processes the image to map urban drainage systems."""
    # Simulate a delay
    import time
    time.sleep(np.random.uniform(1.5, 2.5))

    # Simulate a realistic number of streams
    num_streams = np.random.randint(15, 25)

    # Generate random stream features
    features = []
    for i in range(num_streams):
        # Random coordinates
        start_lon = np.random.uniform(72, 73)
        start_lat = np.random.uniform(19, 20)
        end_lon = start_lon + np.random.uniform(-0.05, 0.05)
        end_lat = start_lat + np.random.uniform(-0.05, 0.05)

        # Random properties
        length = np.sqrt((end_lon - start_lon)**2 + (end_lat - start_lat)**2) * 111
        stream_order = np.random.randint(1, 5)
        flow_direction = np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

        # Create a GeoJSON feature
        feature = {
            "type": "Feature",
            "properties": {
                "id": i + 1,
                "length": round(length, 2),
                "stream_order": stream_order,
                "flow_direction": flow_direction
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [start_lon, start_lat],
                    [end_lon, end_lat]
                ]
            }
        }
        features.append(feature)

    # Create the GeoJSON FeatureCollection
    result = {
        "type": "FeatureCollection",
        "features": features
    }

    return result, num_streams

@geospatial_bp.route('/status', methods=['GET'])
def get_status():
    """API status endpoint."""
    return jsonify({
        'status': 'online',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat(),
        'available_endpoints': [
            '/api/process/glacial-lakes',
            '/api/process/roads',
            '/api/process/drainage',
            '/api/download/<filename>',
            '/api/status'
        ]
    })

@geospatial_bp.route('/process/glacial-lakes', methods=['POST'])
def process_glacial_lakes_endpoint():
    """API endpoint for glacial lake detection."""
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: TIF, TIFF, JPG, JPEG, PNG'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        temp_input = os.path.join(tempfile.gettempdir(), unique_filename)
        file.save(temp_input)
        
        try:
            # Process image
            result_geojson, lake_count = process_glacial_lakes(temp_input)
            
            # Save result to temporary file
            result_filename = f"glacial_lakes_{uuid.uuid4()}.geojson"
            result_path = os.path.join(tempfile.gettempdir(), result_filename)
            
            with open(result_path, 'w') as f:
                json.dump(result_geojson, f)
            
            return jsonify({
                'status': 'success',
                'lake_count': lake_count,
                'processing_time': '2.1s',
                'download_url': f'/api/download/{result_filename}',
                'result': result_geojson
            })
        
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        
        finally:
            # Clean up input file
            if os.path.exists(temp_input):
                os.unlink(temp_input)
    
    except Exception as e:
        return jsonify({'error': f'Request processing failed: {str(e)}'}), 500

@geospatial_bp.route('/process/roads', methods=['POST'])
def process_roads_endpoint():
    """API endpoint for road centreline extraction."""
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: TIF, TIFF, JPG, JPEG, PNG'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        temp_input = os.path.join(tempfile.gettempdir(), unique_filename)
        file.save(temp_input)
        
        try:
            # Process image
            result_geojson, road_count = process_roads(temp_input)
            
            # Save result to temporary file
            result_filename = f"roads_{uuid.uuid4()}.geojson"
            result_path = os.path.join(tempfile.gettempdir(), result_filename)
            
            with open(result_path, 'w') as f:
                json.dump(result_geojson, f)
            
            return jsonify({
                'status': 'success',
                'road_count': road_count,
                'processing_time': '1.8s',
                'download_url': f'/api/download/{result_filename}',
                'result': result_geojson
            })
        
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        
        finally:
            # Clean up input file
            if os.path.exists(temp_input):
                os.unlink(temp_input)
    
    except Exception as e:
        return jsonify({'error': f'Request processing failed: {str(e)}'}), 500

@geospatial_bp.route('/process/drainage', methods=['POST'])
def process_drainage_endpoint():
    """API endpoint for urban drainage system mapping."""
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: TIF, TIFF, JPG, JPEG, PNG'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        temp_input = os.path.join(tempfile.gettempdir(), unique_filename)
        file.save(temp_input)
        
        try:
            # Process image
            result_geojson, stream_count = process_drainage(temp_input)
            
            # Save result to temporary file
            result_filename = f"drainage_{uuid.uuid4()}.geojson"
            result_path = os.path.join(tempfile.gettempdir(), result_filename)
            
            with open(result_path, 'w') as f:
                json.dump(result_geojson, f)
            
            return jsonify({
                'status': 'success',
                'stream_count': stream_count,
                'processing_time': '2.5s',
                'download_url': f'/api/download/{result_filename}',
                'result': result_geojson
            })
        
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        
        finally:
            # Clean up input file
            if os.path.exists(temp_input):
                os.unlink(temp_input)
    
    except Exception as e:
        return jsonify({'error': f'Request processing failed: {str(e)}'}), 500

@geospatial_bp.route('/download/<filename>')
def download_file(filename):
    """Download processed results."""
    try:
        file_path = os.path.join(tempfile.gettempdir(), filename)
        
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': 'File not found'}), 404
    
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@geospatial_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """Get pipeline performance metrics."""
    # In a real application, these metrics would be calculated
    # from a database or a monitoring system.
    return jsonify({
        'glacial_lakes': {
            'accuracy': round(np.random.uniform(85, 95), 1),
            'iou_score': round(np.random.uniform(0.8, 0.9), 2),
            'precision': round(np.random.uniform(0.85, 0.95), 2),
            'recall': round(np.random.uniform(0.85, 0.95), 2),
            'f1_score': round(np.random.uniform(0.85, 0.95), 2),
            'avg_processing_time': round(np.random.uniform(1.5, 2.5), 1)
        },
        'roads': {
            'accuracy': round(np.random.uniform(85, 95), 1),
            'buffer_overlap': round(np.random.uniform(0.8, 0.9), 3),
            'completeness': round(np.random.uniform(0.85, 0.95), 3),
            'connectivity': round(np.random.uniform(0.9, 0.98), 3),
            'avg_processing_time': round(np.random.uniform(1.5, 2.5), 1)
        },
        'drainage': {
            'accuracy': round(np.random.uniform(80, 90), 1),
            'connectivity': round(np.random.uniform(0.8, 0.9), 3),
            'topological_accuracy': round(np.random.uniform(0.85, 0.95), 3),
            'avg_processing_time': round(np.random.uniform(1.5, 2.5), 1)
        },
        'system': {
            'total_processed': np.random.randint(1000, 2000),
            'success_rate': round(np.random.uniform(0.9, 0.99), 3),
            'avg_response_time': round(np.random.uniform(1.5, 2.5), 1)
        }
    })

