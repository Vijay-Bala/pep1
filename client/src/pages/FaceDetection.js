import React, { useState } from 'react';
import axios from 'axios';
import '../css/FaceDetection.css'; // Import custom CSS
import { Link, useNavigate } from 'react-router-dom';

const FaceDetection = ({ totalreg }) => {
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
          <h2 className="card-header">Today's Attendance</h2>
          <br />
          <br />
          <br />
          <br />
          <br />
          <button className="btn btn-primary btn-lg" id='take_attendance' onClick={handleTakeAttendance} style={{marginLeft : "30vw"}}>
            Take Attendance
          </button>
        </div>
      </div>
    </div>
  );
};

export default FaceDetection;
