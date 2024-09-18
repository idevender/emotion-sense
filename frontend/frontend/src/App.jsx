// src/App.jsx

import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [image, setImage] = useState(null);
  const [emotion, setEmotion] = useState('');
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    setEmotion('');
    setImage(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      alert('Please select an image');
      return;
    }
    setLoading(true);

    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await axios.post('http://localhost:5001/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log('Response:', response.data);  // Log response
      setEmotion(response.data.emotion);
    } catch (error) {
      console.error('Error details:', error.response?.data || error.message);
      alert('Error predicting emotion');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mt-5">
      <h1>Emotion Recognition</h1>
      <form onSubmit={handleSubmit}>
        <div className="mb-3">
          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="form-control"
          />
        </div>
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? 'Processing...' : 'Predict Emotion'}
        </button>
      </form>
      {emotion && (
        <div className="mt-4">
          <h3>
            Detected Emotion: <span className="text-success">{emotion}</span>
          </h3>
        </div>
      )}
    </div>
  );
}

export default App;
