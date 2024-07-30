import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useCookies } from "react-cookie";
import axios from "axios";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "bootstrap/dist/css/bootstrap.min.css";
import "./StudentHome.css";
import LeaveForm from "./LeaveForm";
import StudentDetailsComponent from './StudentDetailsComponent';
import DetailsComponent from "./DetailsComponent";
import HodMainComponent from "./HodMainComponent";
import StudentMainComponent from "./StudentMainComponent";
import CouncellorMainComponent from "./CouncellorMainComponent";
import StaffMainComponent from "./StaffMainComponent";
import Preloader from './Preloader';
import LimitReachedPage from "./LimitReachedPage";

const StudentHome = () => {
  const [user, setUser] = useState(null);
  const [activeButton, setActiveButton] = useState("attendance");
  const navigate = useNavigate();
  const [cookies, setCookie, removeCookie] = useCookies([]);
  const arr = ["fullstack", "ml"];

  useEffect(() => {
    const verifyUser = async () => {
      if (!cookies.jwt) {
        navigate("/login");
      } else {
        const { data } = await axios.post(
          "http://localhost:4000",
          {},
          {
            withCredentials: true,
          }
        );
        if (!data.status) {
          removeCookie("jwt");
          navigate("/login");
        } else {
          console.log("data: ", data.user);
          setUser(data.user);
        }
      }
    };
    verifyUser();
  }, [cookies, navigate, removeCookie]);

  const handleButtonClick = (buttonName) => {
    setActiveButton(buttonName);
    if (buttonName === "logout") {
      removeCookie("jwt");
      navigate("/login");
    }
  };

  return (
    <div className="app-container">
      <Preloader />
      <div className="container mt-4">
        <div className="row justify-content-center">
          <div className="col-12 col-lg-15">
            <div className="card shadow p-4">
              <h1 className="text-center mb-10">Attendance Management System</h1>
              <div className="content">
                {user && user.role === "staff" && <StaffMainComponent user={user} domains={[user.domain]} />}
                {user && user.role === "councellor" && <CouncellorMainComponent user={user} />}
                {user && user.role === "hod" && <HodMainComponent user={user} domains={arr} />}
              </div>
            </div>
          </div>
        </div>
      </div>
      <ToastContainer />
    </div>
  );
};

const MessageComponent = ({ user }) => {
  const [requests, setRequests] = useState([]);
  const [requestDetails, setRequestDetails] = useState(null);
  const [individualRequest, setIndividualRequest] = useState(false);
  const [allRequest, setAllRequest] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.post('http://localhost:4000/messagePage', {
          userId: user._id,
        });
        const d = response.data;
        console.log("received Messages: ", d);
        if (d.status) {
          setRequests(d.messages);
        } else {
          console.error('Error:', d.message);
        }
      } catch (error) {
        console.error('Fetch error:', error.message);
      }
    };

    fetchData();
  }, [user]);

  const handleIndividualRequest = (request) => {
    setRequestDetails(request);
    setIndividualRequest(true);
    setAllRequest(false);
  };

  const handleViewPdf = () => {
    if (requestDetails.pdfUrl) {
      const d = Uint8Array.from(requestDetails.pdfUrl.data);
      const blob = new Blob([d], { type: 'application/pdf' });
      const url = URL.createObjectURL(blob);
      window.open(url, '_blank');
    }
  };

  const handleBack = () => {
    setIndividualRequest(false);
    setAllRequest(true);
    window.location.reload();
  };

  return (
    <div className="main-Request">
      {allRequest && (
        <div className="request">
          MessageComponent
          {user && user.name}
          {requests && requests.map((request) => (
            <div className="request-row" key={request._id}>
              <h1>{request && request.result}</h1>
              <button onClick={() => handleIndividualRequest(request)}>Click</button>
              <br />
            </div>
          ))}
        </div>
      )}
      {individualRequest && (
        <div className="individual-Request">
          <h1>{requestDetails && requestDetails.userDetail[0]}</h1>
          <h1>{requestDetails && requestDetails.userDetail[1]}</h1>
          <h1>{requestDetails && requestDetails.userDetail[2]}</h1>
          {requestDetails && requestDetails.messageByCon !== " " && <h1>Councellor: {requestDetails.messageByCon}</h1>}
          {requestDetails && requestDetails.messageByHOD !== " " && <h1>HOD: {requestDetails.messageByHOD}</h1>}
          <h1>Reason: {requestDetails && requestDetails.reason}</h1>
          <h1>{requestDetails && requestDetails.singleDate}</h1>
          <button onClick={handleViewPdf}>View PDF</button>
          <button onClick={handleBack}>Back</button>
        </div>
      )}
    </div>
  );
};

export default StudentHome;
