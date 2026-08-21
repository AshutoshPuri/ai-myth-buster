const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function checkClaim(claim) {
  let response;

  try {
    response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ claim }),
    });
  } catch (error) {
    throw new Error(
      'Unable to connect to the backend. ' +
        'Make sure the FastAPI server is running.'
    );
  }

  let data = {};

  try {
    data = await response.json();
  } catch {
    data = {};
  }

  if (!response.ok) {
    const message = data.detail || 'The backend returned an unexpected error.';
    throw new Error(`Request failed (${response.status}): ${message}`);
  }

  return data;
}
