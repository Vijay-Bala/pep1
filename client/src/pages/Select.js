import React,{ useState,useEffect } from 'react'
import { useCookies } from "react-cookie";
import { useNavigate } from "react-router-dom";

const Select = ({ user }) => {

    const requests = [
        {userDetail:["Saii","CSE","C"],id:"1234"},
        {userDetail:["Vijay","CSE","B"],id:"4567"},
        {userDetail:["Rupesh","CSE","A"],id:"8900"}
    ]
    const [cookies, setCookie, removeCookie] = useCookies([]);
    const navigate = useNavigate();

  const [selectedRequests, setSelectedRequests] = useState([]);

  const handleIndividualRequestClick = (request) => {
    const isSelected = selectedRequests.includes(request._id);
    setSelectedRequests((prevSelected) =>
      isSelected ? prevSelected.filter((id) => id !== request._id) : [...prevSelected, request._id]
    );
  };

  const handleCheckboxChange = (request) => {
    handleIndividualRequestClick(request);
  };

  const handleSelectAllClick = () => {
    const allRequestIds = requests.map((request) => request._id);
    setSelectedRequests(allRequestIds);
  };

  const handleDeselectAllClick = () => {
    setSelectedRequests([]);
  };
  useEffect(()=>{   
    console.log(selectedRequests);
  },[selectedRequests])
  return (
    <div>
      <button onClick={handleSelectAllClick}>Select All</button>
      <button onClick={handleDeselectAllClick}>Deselect All</button>

      <div className="request-list">
        {requests.map((request) => (
          <div className="request-row" key={request._id}>
            <div className="user-details">
              <h1>{request && request.userDetail[0]}</h1>
              <h1>{request && request.userDetail[1]}</h1>
              <h1>{request && request.userDetail[2]}</h1>
            </div>
            <input
              type="checkbox"
              checked={selectedRequests.includes(request._id)}
              onChange={() => handleCheckboxChange(request)}
            />
          </div>
        ))}
      </div>
    </div>
  );
};
export default Select