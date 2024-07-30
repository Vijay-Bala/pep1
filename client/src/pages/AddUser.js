import React, { useState } from 'react';
import axios from 'axios';
import '../css/FaceDetection.css'; // Import custom CSS
import { Link, useNavigate } from 'react-router-dom';

const AddUser = ({ totalreg }) => {
  const [newUserName, setNewUserName] = useState('');
  const [newUserId, setNewUserId] = useState('');
  const navigate = useNavigate();

  const handleTakeAttendance = async () => {
    try {
      await axios.post('http://127.0.0.1:8000/start/');
      navigate(0); // Reload the page
    } catch (error) {
      console.error('Error taking attendance:', error);
    }
  };

  const handleAddUser = async (event) => {
    event.preventDefault();
    try {
      const formData = new FormData();
      formData.append('newusername', newUserName);
      formData.append('newuserid', newUserId);

      await axios.post('http://127.0.0.1:8000/add/', formData);
      navigate(0); // Reload the page
    } catch (error) {
      console.error('Error adding user:', error);
    }
  };

  return (
    <div className="container mt-10 text-center face-detection-container">
      <div className="row text-center">
        <div className="col card">
          <h2 className="card-header">Add New User</h2>
          <form onSubmit={handleAddUser} encType="multipart/form-data">
            <label htmlFor="newusername" className="form-label">
              Enter New User Name
            </label>
            <input
              type="text"
              id="newusername"
              name="newusername"
              className="form-control"
              value={newUserName}
              onChange={(e) => setNewUserName(e.target.value)}
              required
            />

            <label htmlFor="newuserid" className="form-label">
              Enter New User ID
            </label>
            <input
              type="number"
              id="newuserid"
              name="newuserid"
              className="form-control"
              value={newUserId}
              onChange={(e) => setNewUserId(e.target.value)}
              required
            />

            <button type="submit" className="btn btn-dark float-end mt-3" style={{marginLeft : "29vw"}}>
              Add New User
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default AddUser;
