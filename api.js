// API Utility Functions

const API_BASE_URL = '/api';

/**
 * Fetches data from the specified API endpoint.
 * @param {string} endpoint - The API endpoint to fetch data from.
 * @returns {Promise<any>} - A promise that resolves with the JSON response.
 */
async function fetchFromAPI(endpoint) {
    try {
        const response = await fetch(`${API_BASE_URL}/${endpoint}`);
        if (!response.ok) {
            throw new Error(`API request failed with status ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching from API endpoint "${endpoint}":`, error);
        throw error;
    }
}

/**
 * Posts data to the specified API endpoint.
 * @param {string} endpoint - The API endpoint to post data to.
 * @param {object} body - The data to post.
 * @returns {Promise<any>} - A promise that resolves with the JSON response.
 */
async function postToAPI(endpoint, body) {
    try {
        const response = await fetch(`${API_BASE_URL}/${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(body)
        });
        if (!response.ok) {
            throw new Error(`API request failed with status ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Error posting to API endpoint "${endpoint}":`, error);
        throw error;
    }
}

/**
 * Uploads a file to the specified API endpoint.
 * @param {string} endpoint - The API endpoint to upload the file to.
 * @param {File} file - The file to upload.
 * @returns {Promise<any>} - A promise that resolves with the JSON response.
 */
async function uploadFile(endpoint, file) {
    try {
        const formData = new FormData();
        formData.append('image', file);

        const response = await fetch(`${API_BASE_URL}/${endpoint}`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) {
            throw new Error(`File upload failed with status ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Error uploading file to API endpoint "${endpoint}":`, error);
        throw error;
    }
}
