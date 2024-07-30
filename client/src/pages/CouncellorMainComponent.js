import React,{ useState,useEffect } from 'react'
import { useCookies } from "react-cookie";
import { useNavigate } from "react-router-dom";
import moment from 'moment';
import DetailsComponent from './DetailsComponent';
import RequestComponent from './RequestComponent'
import axios from 'axios';
import LeaveTable from './LeaveTable';

const CouncellorMainComponent = ({user}) => {
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
                className={activeButton === "classdetails" ? "active" : ""}
                onClick={() => handleButtonClick("classdetails")}
                >
                Class Details
                </button>

                <button
                className={activeButton === "todayabsent" ? "active" : ""}
                onClick={() => handleButtonClick("todayabsent")}
                >
                Today Absentees 
                </button>

                <button
                className={activeButton === "request" ? "active" : ""}
                onClick={() => handleButtonClick("request")}
                >
                Requests 
                </button>
                
                
                <button
                    className={activeButton === "logout" ? "active" : ""}
                    onClick={() => handleButtonClick("logout")}
                >
                    Log Out
                </button>
            </div> 
            <div className='councellor-toggle'>
            <select id="select-staff" value={selectedStaff} onChange={handleSelectChange}>
                <option value="">Choose Staff</option>
                {staffs.map((staff,index)=> (
                    <option key={index} value={staff}>{staff}</option>
                ))}
            </select>
            </div>
            <div className='content-student'>
                {activeButton==="profile" && <DetailsComponent user={user}/>}
                {activeButton==="request" && <RequestComponent user={user} selectedStaff={selectedStaff}/>}
                {activeButton==="todayabsent" && <CouncellorTodayAbsent user={user}/>}
                {activeButton==="classdetails" && <CouncellorClassDetails user={user}/>}

            </div>
        </div>
    )
}




const CouncellorTodayAbsent = ({user}) => {
    const [selectedDay, setSelectedDay] = useState(''); // Selected day
    const [classInfo, setClassInfo] = useState({}); // Student information for the selected day
    const [studentInfo, setStudentInfo] = useState([]);
    const [selectedRegisnos, setSelectedRegisnos] = useState([]);
    const [showStudents,setShowStudents] = useState(false);
    const [allRegisno,setAllRegisno] = useState([]);
    const classArr=(user.name).split(' ')
  
  
    const fetchStudentInfo = async (day) => {
      try {
        console.log(day);
        const response = await axios.post(`http://localhost:4000/studentInfo?day=${day}`,{
          batch:user.batch,
          section:classArr[1],
          depart:classArr[0],
        });
        setClassInfo(response.data.studentInfo);
        setStudentInfo(response.data.studentInfo.students);
        setAllRegisno(response.data.allRegisno);
        
        if(response.data.studentInfo && response.data.studentInfo.enrolled===false){
          setShowStudents(true);
        }else{
          setShowStudents(false);
        }
      } catch (error) {
        console.error('Error fetching student information:', error);
      }
    };
  
    const handleDayClick = (e) => {
      console.log("selected day: ",e.target.value);
      setSelectedDay(e.target.value);
      fetchStudentInfo(e.target.value);
    };
  
    const handleCheckboxChange = (regisno) => {
      const updatedRegisnos = selectedRegisnos.includes(regisno)
        ? selectedRegisnos.filter((r) => r !== regisno)
        : [...selectedRegisnos, regisno];
  
      setSelectedRegisnos(updatedRegisnos);
      // allRegisno=allRegisno.filter((regis) => !updatedRegisnos.includes(regis))
      console.log(updatedRegisnos);
      console.log(allRegisno);
    };

    const handleAbsentees = async() => {
      try{
        const updatedTotal = allRegisno.filter((element) => !selectedRegisnos.includes(element));
        const response = await axios.post('http://localhost:4000/handleAbsentees',{
          absentees:selectedRegisnos,
          presentees:updatedTotal,
          depart:classArr[0],
          section:classArr[1],
          date:selectedDay,
          batch:user.batch
        })
        if(response.data.status===true){
          setShowStudents(false)
        }
      }catch(err){
        console.log(err.message);
      }
    }
    const handleEdit = async () => {
      console.log("selected date: ",selectedDay);
      try{
        const response = await axios.post(`http://localhost:4000/getAbsentees?day=${selectedDay}`,{
          depart:classArr[0],
          batch:user.batch,
          section:classArr[1]
        });
        if(response.data.status===true){
          console.log("absent: ",response.data.regisnos);
          console.log("total: ",allRegisno);
          setSelectedRegisnos(response.data.regisnos)
        }
      }catch(err){
        console.log(err.message);
      }
      setShowStudents(true);
    }
    return (
      <div>
        <h1>Attendance Page</h1>
        {((selectedDay=='') || (selectedDay && !showStudents) )&& <input type='date' value={selectedDay} onChange={handleDayClick}/>}
        {selectedDay && showStudents &&(
          <div>
            <h2>Student Information for {selectedDay}</h2>
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>RollNo</th>
                  <th>RegisterNo</th>
                  <th>Make Absent</th>
                </tr>
              </thead>
              <tbody>
                  {studentInfo.map((student) => (
                  <tr key={student.rollno}>
                    <td>{student.name}</td>
                    <td>{student.rollno}</td>
                    <td>{student.regisno}</td>
                    <td>
                    <input
                      type="checkbox"
                      checked={selectedRegisnos.includes(student.regisno)}
                      onChange={() => handleCheckboxChange(student.regisno)}
                    />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <button onClick={handleAbsentees}>Submit</button>
          </div>
        )}
        {!showStudents && selectedDay &&<button onClick={handleEdit}>Edit</button>}
      </div>
    );
  };






const CouncellorClassDetails = ({user}) => {
  const [data,setData]=useState({})
  const classArr=(user.name).split(' ')

  useEffect(()=>{
    const fetchDetails = async () => {
      try{
        console.log("hello");
        const response = await axios.post("http://localhost:4000/classDetails",{
          section:classArr[1],
          depart:classArr[0],
          batch:user.batch
        })
        if(response.data.status===true){
          setData(response.data.result);
          console.log(response.data.result);
        }
      }catch(err){
        console.log(err);
      }
    }
    fetchDetails()
  },[])

  return (
      <div>CouncellorClassDetails
          <LeaveTable info={data}/>
      </div>
  )
}

export default CouncellorMainComponent