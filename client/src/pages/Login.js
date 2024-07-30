import React, { useState, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { useCookies } from "react-cookie";
import { ToastContainer, toast } from "react-toastify";
import 'bootstrap/dist/css/bootstrap.min.css';
import 'react-toastify/dist/ReactToastify.css';
import '../css/login.css'; // Import the custom CSS file
import Preloader from './Preloader';

function Login() {
  const [cookies] = useCookies([]);
  const [selectedDomainOption, setSelectedDomainOption] = useState('');
  const [selectedRoleOption, setSelectedRoleOption] = useState('');
  const [password, setPassword] = useState('');

  const navigate = useNavigate();

  const handleChange1 = (event) => {
    setSelectedRoleOption(event.target.value);
  };

  const handleChange2 = (event) => {
    setSelectedDomainOption(event.target.value);
  };

  useEffect(() => {
    if (cookies.jwt) {
      navigate("/");
    }
  }, [cookies, navigate]);

  const generateError = (error) =>
    toast.error(error, {
      position: "bottom-right",
    });

  const handleSubmit = async (event) => {
    console.log(selectedDomainOption, selectedRoleOption, password);
    event.preventDefault();
    try {
      const { data } = await axios.post(
        "http://localhost:4000/login",
        {
          role: selectedRoleOption,
          domain: (selectedRoleOption !== "hod" ? selectedDomainOption : 'hod'),
          password: password
        },
        { withCredentials: true }
      );
      if (data) {
        if (data.errors) {
          const { regis, password } = data.errors;
          if (regis) generateError(regis);
          else if (password) generateError(password);
        } else {
          navigate("/");
        }
      }
    } catch (ex) {
      console.log(ex);
    }
  };

  return (
    <div>
    <Preloader />
      <h1 className="text-center mb-4" id="main_h">Attendance Management System</h1>
      <div className="container">
        <div className="row justify-content-center">
          <div className="col-md-5 form-container">
            <h2 className="text-center mb-4">Login to your Account</h2>
            <form onSubmit={(e) => handleSubmit(e)}>
              <div className="form-group mb-3">
                <label htmlFor="role">Role</label>
                <select 
                  id="dropdown" 
                  className="form-control" 
                  value={selectedRoleOption} 
                  onChange={handleChange1}
                >
                  <option value="">Select...</option>
                  <option value="staff">Staff</option>
                  <option value="admin">Admin</option>
                  <option value="hod">HOD</option>
                  {/* Add more options as needed */}
                </select>
              </div>
              {selectedRoleOption !== "hod" && (
                <div className="form-group mb-3">
                  <label htmlFor="domain">Domain</label>
                  <select 
                    id="dropdown" 
                    className="form-control" 
                    value={selectedDomainOption} 
                    onChange={handleChange2}
                  >
                    <option value="">Select...</option>
                    <option value="fullstack">Fullstack</option>
                    <option value="ml">ML</option>
                    {/* Add more options as needed */}
                  </select>
                </div>
              )}
              <div className="form-group mb-3">
                <label htmlFor="password">Password</label>
                <input
                  type="password"
                  className="form-control"
                  placeholder="Password"
                  name="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>
              <button type="submit" className="btn btn-primary w-100">Submit</button>
            </form>
          </div>
        </div>
        <ToastContainer />
      </div>
    </div>
  );
}

export default Login;
