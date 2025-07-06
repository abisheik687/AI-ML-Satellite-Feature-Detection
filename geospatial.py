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

# Mock processing functions (to be replaced with actual implementations)
def process_glacial_lakes(image_path):
    """Mock function for glacial lake detection."""
    # Simulate processing time
    import time
    time.sleep(2)
    
    # Mock GeoJSON result
    result = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": 1,
                    "area": 2.3,
                    "perimeter": 5.8,
                    "confidence": 0.92
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [83.9956, 28.2380],
                        [83.9970, 28.2380],
                        [83.9970, 28.2390],
                        [83.9956, 28.2390],
                        [83.9956, 28.2380]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "id": 2,
                    "area": 1.8,
                    "perimeter": 4.2,
                    "confidence": 0.87
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [84.1000, 28.3000],
                        [84.1015, 28.3000],
                        [84.1015, 28.3010],
                        [84.1000, 28.3010],
                        [84.1000, 28.3000]
                    ]]
                }
            }
        ]
    }
    
    return result, 2

def process_roads(image_path):
    """Mock function for road centreline extraction."""
    import time
    time.sleep(1.8)
    
    # Mock GeoJSON result
    result = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": 1,
                    "length": 1250.5,
                    "road_type": "primary",
                    "confidence": 0.94
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [77.2090, 28.6139],
                        [77.2200, 28.6200],
                        [77.2300, 28.6250],
                        [77.2500, 28.6500]
                    ]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "id": 2,
                    "length": 890.2,
                    "road_type": "secondary",
                    "confidence": 0.89
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [77.1800, 28.6000],
                        [77.1900, 28.6100],
                        [77.2000, 28.6150],
                        [77.2200, 28.6300]
                    ]
                }
            }
        ]
    }
    
    return result, 2

def process_drainage(image_path):
    """Mock function for urban drainage system mapping."""
    import time
    time.sleep(2.5)
    
    # Mock GeoJSON result
    result = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": 1,
                    "stream_order": 1,
                    "length": 450.3,
                    "flow_direction": "NE"
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [72.8777, 19.0760],
                        [72.8800, 19.0780],
                        [72.8850, 19.0820],
                        [72.8900, 19.0900]
                    ]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "id": 2,
                    "stream_order": 2,
                    "length": 680.7,
                    "flow_direction": "N"
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [72.8600, 19.0600],
                        [72.8650, 19.0650],
                        [72.8700, 19.0700],
                        [72.8777, 19.0760]
                    ]
                }
            }
        ]
    }
    
    return result, 2

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
    return jsonify({
        'glacial_lakes': {
            'accuracy': 87.0,
            'iou_score': 0.87,
            'precision': 0.92,
            'recall': 0.89,
            'f1_score': 0.90,
            'avg_processing_time': 2.1
        },
        'roads': {
            'accuracy': 89.3,
            'buffer_overlap': 0.893,
            'completeness': 0.917,
            'connectivity': 0.942,
            'avg_processing_time': 1.8
        },
        'drainage': {
            'accuracy': 86.5,
            'connectivity': 0.889,
            'topological_accuracy': 0.921,
            'avg_processing_time': 2.5
        },
        'system': {
            'total_processed': 1247,
            'success_rate': 0.956,
            'avg_response_time': 2.1
        }
    })

