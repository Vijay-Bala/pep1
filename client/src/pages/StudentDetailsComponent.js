import React, { useEffect, useState } from "react";
import axios from "axios";
import * as XLSX from 'xlsx';
import '../css/StudentDetailsComponent.css'

const StudentDetailsComponent = ({user}) => {

    const [studentDetails,setStudentDetails] = useState(null);
    const [selectedClass,setSelectedClass] = useState('')
    const AvailableClasses = user.classes;
  
    useEffect(() => {
      const fetchData = async () => {
        try {
          const response = await axios.post('http://localhost:4000/fetchStudentDetails',{
            class:(selectedClass!="all")?[selectedClass]:AvailableClasses
          });
  
          const d = response.data;
          console.log("received Student Details: ",d);
          if (d.status) {
            // If the response is successful, you can set the PDF data
            setStudentDetails(d.details);
          } else {
            // Handle errors
            console.error('Error:', d.message);
          }
        } catch (error) {
          console.error('Fetch error:', error.message);
        }
      };
  
      fetchData();
    }, [selectedClass,user]);

    const exportToExcel = () => {
      // Create a new Excel workbook
      const modifiedData = studentDetails.map(({ classes, ...rest }) => rest);

      const wb = XLSX.utils.book_new();
  
      // Convert your table data to an Excel sheet
      const ws = XLSX.utils.json_to_sheet(modifiedData);
  
      // Add the sheet to the workbook
      XLSX.utils.book_append_sheet(wb, ws, 'Sheet1');
  
      // Save the workbook to a file
      XLSX.writeFile(wb, 'details.xlsx');
    };

    const handleClassOption = (e) => {
        setSelectedClass(e.target.value)
    }
    return (
    <div className="studentDetails-component">
        Filter: 

        <select value={selectedClass} onChange={handleClassOption}>
            <option value="">Select Class</option>
            <option value="all">All</option>
            {AvailableClasses.map((individualClass) => (
              <option key={individualClass} value={individualClass}>{individualClass}</option>
            ))}
        </select>
        {
            studentDetails && <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Department</th>
                  <th>Section</th>
                  <th>Register No</th>
                  <th>Total Leave</th>
                </tr>
              </thead>
              <tbody>
                {studentDetails.map((item, index) => (
                  <tr key={index}>
                    <td>{item.name}</td>
                    <td>{item.depart}</td>
                    <td>{item.section}</td>
                    <td>{item.regisno}</td>
                    <td>{item.leaves}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {studentDetails.length!=0 && <button onClick={exportToExcel}>Download</button>}
          </div>
        }
    </div>)
  };

export default StudentDetailsComponent;