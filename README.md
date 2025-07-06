# AI-ML-Satellite-Feature-Detection

## Description

This project is a Flask-based web application designed for geospatial image processing. It leverages AI/ML models (currently mocked) to detect various features from satellite imagery, such as glacial lakes, roads, and urban drainage systems. The application provides a set of RESTful APIs to upload images, process them, and retrieve the results in GeoJSON format. It also includes basic user management functionalities.

## Features

-   **Glacial Lake Detection:** API endpoint for identifying and outlining glacial lakes from satellite images.
-   **Road Centerline Extraction:** API endpoint for extracting road centrelines from satellite imagery.
-   **Urban Drainage System Mapping:** API endpoint for mapping urban drainage systems.
-   **File Upload:** Supports uploading images in TIFF, JPG, JPEG, and PNG formats for processing.
-   **GeoJSON Output:** Processing results are provided in a standardized GeoJSON format.
-   **API Endpoints:** Offers a clear and easy-to-use API for all functionalities.
-   **User Management:** Basic CRUD (Create, Read, Update, Delete) operations for users.
-   **Status and Metrics:** Endpoints to check API status and mock performance metrics.
-   **Static Frontend:** Serves a basic `index.html` for interaction (details in `static/` folder).

## Technologies Used

-   **Backend:** Python, Flask
-   **API Development:** Flask Blueprints
-   **Database:** SQLite (with SQLAlchemy ORM for user management)
-   **CORS:** Flask-CORS for cross-origin requests.
-   **Geospatial Data Handling:** (The current processing functions are mocks. In a real implementation, libraries like GDAL, Rasterio, Shapely, Fiona, PyProj would be used).
-   **Dependencies**: `Flask`, `Flask-CORS`, `Flask-SQLAlchemy`, `Werkzeug`, `Numpy` (see `requirements.txt`)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd AI-ML-Satellite-Feature-Detection
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Directory Structure for `main.py`:**
    The `main.py` script includes `sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))`.
    - If `main.py` is at the project root (e.g., `AI-ML-Satellite-Feature-Detection/main.py`) and `src` is also at the root (`AI-ML-Satellite-Feature-Detection/src/`), this line attempts to add the grandparent directory of `main.py` to the path. This might be unnecessary if `src` is intended to be a standard Python package importable from the root.
    - If `main.py` is intended to be in a subdirectory (e.g., `AI-ML-Satellite-Feature-Detection/app/main.py`) and `src` is at the project root (`AI-ML-Satellite-Feature-Detection/src/`), then this line correctly adds the project root to `sys.path`, allowing `from src...` imports.

    For this guide, we'll assume `main.py` is at the project root and the `sys.path.insert` line might need adjustment or removal if it causes import issues for `src`. If you encounter `ModuleNotFoundError` for `src`, verify your project structure and this path manipulation.

5.  **Initialize the database:**
    The application automatically creates the `database/app.db` file and the necessary tables when `main.py` is first run. Ensure the `database` directory can be created by the script in the same location as `main.py`.

6.  **Run the application:**
    Navigate to the directory containing `main.py`.
    ```bash
    python main.py
    ```
    The application will be accessible at `http://localhost:5000`.

## API Endpoints

The base URL for all API endpoints is `/api`.

### Geospatial Processing (routes defined in `src/routes/geospatial.py`)

-   `GET /api/status`: Get the status of the API, version, and available endpoints.
-   `POST /api/process/glacial-lakes`: Upload an image for glacial lake detection.
    -   **Request Body:** `multipart/form-data` with an `image` file (TIF, TIFF, JPG, JPEG, PNG).
    -   **Response:** JSON with processing status, mock lake count, download URL for GeoJSON results, and the GeoJSON result itself.
-   `POST /api/process/roads`: Upload an image for road centerline extraction.
    -   **Request Body:** `multipart/form-data` with an `image` file.
    -   **Response:** JSON with processing status, mock road count, download URL for GeoJSON results, and the GeoJSON result.
-   `POST /api/process/drainage`: Upload an image for urban drainage system mapping.
    -   **Request Body:** `multipart/form-data` with an `image` file.
    -   **Response:** JSON with processing status, mock stream count, download URL for GeoJSON results, and the GeoJSON result.
-   `GET /api/download/<filename>`: Download the processed GeoJSON file (files are stored temporarily).
-   `GET /api/metrics`: Get mock performance metrics for the processing pipelines.

### User Management (routes defined in `src/routes/user.py`)

-   `GET /api/users`: Get a list of all users.
-   `POST /api/users`: Create a new user.
    -   **Request Body (JSON):** `{ "username": "newuser", "email": "user@example.com" }`
-   `GET /api/users/<int:user_id>`: Get details of a specific user.
-   `PUT /api/users/<int:user_id>`: Update details of a specific user.
    -   **Request Body (JSON):** `{ "username": "updateduser", "email": "newemail@example.com" }` (fields are optional)
-   `DELETE /api/users/<int:user_id>`: Delete a specific user.

## Usage

1.  **Start the Flask application** as described in the Setup section.
2.  **Use an API client** (like Postman, Insomnia, curl, or the provided `static/index.html`) to interact with the API endpoints.

    **Example: Processing an image for glacial lakes using curl**
    ```bash
    curl -X POST -F "image=@/path/to/your/image.tif" http://localhost:5000/api/process/glacial-lakes
    ```
    The response will contain a `download_url` (e.g., `/api/download/glacial_lakes_xxxx.geojson`) and the `result` (GeoJSON content).

    **Example: Fetching results using the download URL**
    ```bash
    curl http://localhost:5000/api/download/glacial_lakes_xxxx.geojson -o results.geojson
    ```

## Project Structure

(Assuming `main.py` and `src/` are at the project root)
```
.
├── main.py                     # Main Flask application script, entry point
├── requirements.txt            # Python dependencies
├── src/
│   ├── models/
│   │   └── user.py             # SQLAlchemy User model and db instance
│   └── routes/
│       ├── geospatial.py       # Blueprint and API logic for geospatial processing
│       └── user.py             # Blueprint and API logic for user management
├── database/                   # Directory for the SQLite database (created automatically by main.py)
│   └── app.db
├── static/                     # Folder for static assets served by Flask
│   ├── index.html              # Main HTML file for a basic frontend
│   ├── script.js               # JavaScript for the frontend
│   └── styles.css              # CSS for the frontend
├── architecture_diagram.mmd    # Mermaid diagram for detailed architecture
├── architecture_diagram_simple.mmd # Mermaid diagram for simplified architecture
├── architecture_diagram_simple.png # Rendered simplified architecture diagram
├── process_flow_diagram.mmd    # Mermaid diagram for process flow
├── process_flow_diagram.png    # Rendered process flow diagram
├── README.md                   # This file
└── *.md                        # Other Markdown files for documentation
                                # (e.g., Deployment Package and Installation Guide.md)
```

## Notes and Future Considerations

-   **Actual AI/ML Implementation:** The current geospatial processing functions in `src/routes/geospatial.py` are mock implementations. These need to be replaced with actual model loading and inference logic.
-   **Error Handling:** Enhance error handling throughout the application for robustness.
-   **Configuration Management:** Externalize configurations (e.g., `SECRET_KEY`, `MAX_FILE_SIZE`, database URI) using environment variables or configuration files (e.g., `.env`, `config.py`).
-   **Testing:** Implement comprehensive unit and integration tests for all modules.
-   **Asynchronous Processing:** For potentially long-running image processing tasks, integrate a task queue like Celery with a message broker (e.g., Redis, RabbitMQ).
-   **Security:** Conduct a security review. Sanitize inputs, protect against common web vulnerabilities (XSS, CSRF, SQLi), and manage secrets properly.
-   **Frontend:** The `static/index.html` is basic. A more interactive frontend framework (e.g., React, Vue, Angular) could be used for a richer user experience.
-   **API Documentation:** Generate interactive API documentation using tools like Swagger/OpenAPI.
-   **Pathing in `main.py`:** Carefully review the `sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))` line in `main.py`. Ensure it correctly resolves the `src` directory based on your chosen project layout and how `main.py` is executed.