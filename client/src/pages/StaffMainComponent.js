import React, { useState, useEffect } from 'react';
import { useCookies } from "react-cookie";
import axios from 'axios';
import DetailsComponent from './DetailsComponent';
import HodDepartmentDetails from './HodDepartmentDetails';
import CustomDatePicker from './CustomDatePicker';
import FaceDetection from './FaceDetection'; // Import the FaceDetection component
import '../css/StaffMainComponent.css'; // Import custom CSS
import AddUser from './AddUser';

// Error Boundary component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error", error, errorInfo);
  }

  componentDidUpdate(prevProps, prevState) {
    if (prevState.hasError !== this.state.hasError && this.state.hasError) {
      window.location.reload(); // Reload the page on error
    }
  }

  render() {
    return this.props.children; 
  }
}

const StaffMainComponent = ({ user, domains }) => {
  const [cookies, setCookie, removeCookie] = useCookies([]);
  const [activeButton, setActiveButton] = useState("profile");

  const handleButtonClick = (buttonName) => {
    if (buttonName === "logout") {
      removeCookie("jwt");
      window.location.href = "/login"; // Redirect to login page
    } else {
      setActiveButton(buttonName);
    }
  };

  return (
    <ErrorBoundary>
      <div className='student-main'>
        <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
          <div className="container-fluid">
            <button
              className={`btn btn-link ${activeButton === "profile" ? "active" : ""}`}
              onClick={() => handleButtonClick("profile")}
            >
              Profile
            </button>
            <button
              className={`btn btn-link ${activeButton === "facedetection" ? "active" : ""}`}
              onClick={() => handleButtonClick("facedetection")}
            >
              Face Detection
            </button>
            <button
              className={`btn btn-link ${activeButton === "adduser" ? "active" : ""}`}
              onClick={() => handleButtonClick("adduser")}
            >
              Add User
            </button>
            <button
              className={`btn btn-link ${activeButton === "leaveDetails" ? "active" : ""}`}
              onClick={() => handleButtonClick("leaveDetails")}
            >
              Leave Details
            </button>
            <button
              className={`btn btn-link ${activeButton === "attendance" ? "active" : ""}`}
              onClick={() => handleButtonClick("attendance")}
            >
              Attendance
            </button>
            <button
              className="btn btn-link"
              onClick={() => handleButtonClick("logout")}
            >
              Log Out
            </button>
          </div>
        </nav>
        
        <div className='content-student container-fluid mt-4'>
          {activeButton === "profile" && <DetailsComponent user={user} />}
          {activeButton === "facedetection" && <FaceDetection user={user} />}
          {activeButton === "adduser" && <AddUser user={user} />}
          {activeButton === "attendance" && <Attendance user={user} />}
          {activeButton === "leaveDetails" && <HodDepartmentDetails user={user} domains={domains} />}
        </div>
      </div>
    </ErrorBoundary>
  );
};

const Attendance = ({ user }) => {
  const [students, setStudents] = useState([]);
  const [attendanceId, setAttendanceId] = useState(null);
  const [date, setDate] = useState(null);

  const fetchAttendance = async () => {
    if (!date) return;

    const formattedDate = new Date(date).toISOString().split('T')[0]; // Format the date as yyyy-MM-dd
    try {
      const response = await axios.get(
        `http://localhost:4000/attendance?date=${formattedDate}&domain=${user.domain}&batch=${user.batch}`
      );
      if (response.data) {
        if (Object.keys(response.data).length === 0) {
          setDate(null)
          alert('Pick the Valid Date!!!')
          const date = prompt('Enter the date as (yyyy-MM-dd)')
          await axios.post('http://localhost:4000/insertDate', {
            date: date,
            batch: user.batch,
            domain: user.domain
          })
        } else {
          setStudents(response.data.students);
          setAttendanceId(response.data._id);
        }
      } else {
        setStudents([]);
        setAttendanceId(null);
      }
    } catch (error) {
      console.error('Failed to fetch attendance:', error);
    }
  };

  const handleDateChange = (selectedDate) => {
    setDate(selectedDate);
    setStudents([]);
    setAttendanceId(null);
  };

  const handleCheckboxChange = (index, session, present) => {
    const updatedStudents = [...students];
    updatedStudents[index][session] = present;
    setStudents(updatedStudents);
  };

  const handleReasonChange = (index, session, reason) => {
    const updatedStudents = [...students];
    updatedStudents[index][session] = reason;
    setStudents(updatedStudents);
  };

  const handleSubmit = async () => {
    try {
      if (attendanceId) {
        await axios.put(`http://localhost:4000/attendance/${attendanceId}`, { students });
      } else {
        await axios.post(`http://localhost:4000/attendance`, {
          date: date,
          domain: user.domain,
          batch: user.batch,
          students: students,
        });
      }
      setDate(null);
      setStudents([]);
      setAttendanceId(null);
    } catch (error) {
      console.error('Failed to submit attendance:', error);
    }
  };

  const convertToCSV = () => {
    const header = ['Name', "Register No", 'Roll No', 'Forenoon Present', 'Forenoon Reason', 'Afternoon Present', 'Afternoon Reason'];
    const rows = students.map(student => [
      student.name,
      student.regisno,
      student.rollno,
      student.forePresent ? 'Present' : 'Absent',
      student.foreDetail,
      student.afterPresent ? 'Present' : 'Absent',
      student.afterDetail
    ]);

    const csvContent = header.join(',') + '\n' + rows.map(row => row.join(',')).join('\n');
    const encodedUri = encodeURI('data:text/csv;charset=utf-8,' + csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'students.csv');
    document.body.appendChild(link);
    link.click();
  };

  useEffect(() => {
    console.log("students: ", students);
  }, [students]);
  useEffect(() => {
    console.log("date: ", date);
  }, [date]);

  return (
    <div>
      <CustomDatePicker onDateChange={handleDateChange} date={date} />
      <button onClick={fetchAttendance} className="btn btn-primary" style={{marginBottom : "4vh"}}>Fetch Attendance</button>

      {students.length > 0 && (
        <div>
          <table className="table mt-4">
            <thead>
              <tr>
                <th>Name</th>
                <th>Register No</th>
                <th>Roll No</th>
                <th>Forenoon</th>
                <th>Forenoon Reason</th>
                <th>Afternoon</th>
                <th>Afternoon Reason</th>
              </tr>
            </thead>
            <tbody>
              {students.map((student, index) => (
                <tr key={index}>
                  <td>{student.name}</td>
                  <td>{student.regisno}</td>
                  <td>{student.rollno}</td>
                  <td>
                    <input
                      style={{ marginRight: "15px" }}
                      type="checkbox"
                      checked={student.forePresent}
                      onChange={(e) =>
                        handleCheckboxChange(index, 'forePresent', e.target.checked)
                      }
                    />
                    {students[index].forePresent ? "Present" : "Absent"}
                  </td>
                  <td>
                    <input
                      type="text"
                      value={student.foreDetail}
                      onChange={(e) =>
                        handleReasonChange(index, 'foreDetail', e.target.value)
                      }
                      className="form-control"
                    />
                  </td>
                  <td>
                    <input
                      style={{ marginRight: "15px" }}
                      type="checkbox"
                      checked={student.afterPresent}
                      onChange={(e) =>
                        handleCheckboxChange(index, 'afterPresent', e.target.checked)
                      }
                    />
                    {students[index].afterPresent ? "Present" : "Absent"}
                  </td>
                  <td>
                    <input
                      type="text"
                      value={student.afterDetail}
                      onChange={(e) =>
                        handleReasonChange(index, 'afterDetail', e.target.value)
                      }
                      className="form-control"
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <button onClick={handleSubmit} className="btn btn-success">Submit</button>
        </div>
      )}
      {students.length > 0 && <button onClick={convertToCSV} className="btn btn-secondary mt-2">Download CSV</button>}
    </div>
  );
};

export default StaffMainComponent;
