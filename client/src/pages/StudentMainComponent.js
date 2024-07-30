import React,{ useState,useEffect } from 'react'
import { useCookies } from "react-cookie";
import { useNavigate } from "react-router-dom";
import DetailsComponent from './DetailsComponent';
import axios from 'axios';
import Preloader from './Preloader';
import LimitReachedPage from './LimitReachedPage'

const StudentMainComponent = ({user}) => {
    const [cookies, setCookie, removeCookie] = useCookies([]);
    const [leaveIndividual,setLeaveIndividual] = useState(null)
    const [activeButton, setActiveButton] = useState("profile");
    const navigate = useNavigate();
    const handleButtonClick = (buttonName) => {
        setActiveButton(buttonName);
        if (buttonName === "logout") {
          removeCookie("jwt");
          navigate("/login");
        }
    };
    useEffect(()=>{
        const fetchDetails = async () => {
            const response = await axios.post("http://localhost:4000/leaveIndividual",{
                name:user.name,
                section:user.section,
                depart:user.depart,
                batch:user.batch
            })
            setLeaveIndividual(response.data.result);
            console.log("result: ",response.data.result);
        }
        fetchDetails()
    },[])
    return (
        <div className='student-main' >
        <Preloader />
            <div className="navbar navbar-student" style={{marginBottom: "30px"}}>
                <button
                className={activeButton === "profile" ? "active" : ""}
                onClick={() => handleButtonClick("profile")}
                >
                Profile
                </button>

                <button
                className={activeButton === "leavedetails" ? "active" : ""}
                onClick={() => handleButtonClick("leavedetails")}
                >
                Leave Details
                </button>

                <button
                className={activeButton === "applyleave" ? "active" : ""}
                onClick={() => handleButtonClick("applyleave")}
                >
                Apply Leave
                </button>
                
                <button
                    className={activeButton === "logout" ? "active" : ""}
                    onClick={() => handleButtonClick("logout")}
                >
                    Log Out
                </button>
            </div> 
            <div className='content-student'>
                {activeButton==="leavedetails" && <StudentLeaveDetails user={user} leaveIndividual={leaveIndividual}/>}
                {activeButton==="profile" && <DetailsComponent user={user}/>}
               

            </div>
        </div>
    )
}


const StudentLeaveDetails = ({user,leaveIndividual}) => {
    return (
        <div>StudentLeaveDetails
            <table>
                <thead>
                    <tr>
                        <th>S No</th>
                        <th>Leave Date</th>
                    </tr>
                </thead>
                <tbody>
                   {leaveIndividual && leaveIndividual.map((leave,index)=>(
                    <tr key={index} >
                        <td>{index+1}</td>
                        <td>{leave}</td>
                    </tr>
                   ))}
                </tbody>
            </table>
        </div>
    )
}
export default StudentMainComponent





