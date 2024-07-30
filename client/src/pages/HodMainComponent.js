import React,{ useState,useEffect } from 'react'
import { useCookies } from "react-cookie";
import { useNavigate } from "react-router-dom";
import DetailsComponent from './DetailsComponent';
import HodDepartmentDetails from './HodDepartmentDetails';
import axios from 'axios';
import HodRequestComponent from './HodRequestComponent';

const HodMainComponent = ({user,domains}) => {
    const [cookies, setCookie, removeCookie] = useCookies([]);
    const [activeButton, setActiveButton] = useState("request");
    const [selectedStaff, setSelectedStaff] = useState('');
    const staffs = user.staffs;
    const navigate = useNavigate();
    
    const handleButtonClick = (buttonName) => {
        setActiveButton(buttonName);
        if (buttonName === "logout") {
          removeCookie("jwt");
          navigate("/login");
        }
      };
      

    const handleSelectChange = (e) => {
        setSelectedStaff(e.target.value);
    };
    return (
        <div className='student-main'>
            <div className="navbar navbar-councellor">
                <button
                className={activeButton === "profile" ? "active" : ""}
                onClick={() => handleButtonClick("profile")}
                >
                Profile
                </button>

                <button
                className={activeButton === "leaveDetails" ? "active" : ""}
                onClick={() => handleButtonClick("leaveDetails")}
                >
                Leave Details
                </button>
              
                
                <button
                    className={activeButton === "logout" ? "active" : ""}
                    onClick={() => handleButtonClick("logout")}
                >
                    Log Out
                </button>
            </div> 
          
            <div className='content-student'>
                {activeButton==="profile" && <DetailsComponent user={user}/>}
                {activeButton==="leaveDetails" && <HodDepartmentDetails domains={domains} user={user}/>}

            </div>
        </div>
    )
}





export default HodMainComponent
