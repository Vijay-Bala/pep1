import React, { useState } from 'react';
import DatePicker from 'react-datepicker';
import { format } from 'date-fns';
import axios from 'axios';
import FileUploader from './FileUploader';
import 'react-datepicker/dist/react-datepicker.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import './LeaveForm.css';


const LeaveForm = ({user}) => {
  // leave form states
  const [reason, setReason] = useState('');
  const [leaveType, setLeaveType] = useState('');
  const [isSingleDate, setIsSingleDate] = useState(true);
  const [selectedDate, setSelectedDate] = useState(null);
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);

  // file uploader states
  const [files, setFiles] = useState([]);
  const [fileName, setFileName] = useState('');
  const [choosedFile, setChoosedFile] = useState('');

  const handleSubmit = async () => {    
    console.log(user);
    if(reason==='') {
      alert("Fill the reason");
      return;
    }
    if(leaveType==='') {
      alert("Fill the leaveType");
      return ;
    }
    if((isSingleDate && !selectedDate) || (!selectedDate && !startDate && !endDate) || 
      (!isSingleDate && !startDate && !endDate) ){
      alert("Fill the Dates");
      return;
    }
    try{

      const formData = new FormData();
      files.forEach((fileObj, index) => {
        formData.append(`files`, fileObj.file);
        formData.append(`fileDetails`, fileObj.fileName);
      });
        const res = await axios.post("http://localhost:4000/input",{
          userId:user._id,
          reason:reason,
          depart:user.depart,
          section:user.section,
          selectedDate:isSingleDate? format(selectedDate,'dd.MM.yyyy'):" ",
          startDate:isSingleDate?" ": format(startDate,'dd.MM.yyyy'),
          endDate:isSingleDate?" ": format(endDate,'dd.MM.yyyy'),
          leaveType:leaveType
        })
        const responseData = res.data;

        if (responseData.status === true) {
          
          formData.append('commonId',responseData.requestId);

          await axios.post('http://localhost:4000/upload', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          });

          // Clear the files state for the next upload
          setFiles([]);
          alert("Request successful");
          setReason('')
          setSelectedDate('')
          setStartDate('')
          setEndDate('')
          setLeaveType('')
          // Handle success case here
        } else {
            alert("Request failed");
            // Handle failure case here
        }
    }catch(err){
        console.log(err);
    }

    // console.log('userID:', userId);
    console.log('Reason:', reason);
    console.log('Is Single Date:', isSingleDate);
    console.log('Selected Date:', selectedDate);
    console.log('Start Date:', startDate);
    console.log('End Date:', endDate);
  };
  const renderFileList = () => {
    return (
      <div>
        <h2>Selected Files Information</h2>
        {files.map((fileObj, index) => (
          <div key={index}>
            <p>Information: {fileObj.fileName}</p>
            <p>Original Name: {fileObj.file.name}</p>
            {/* <p>MIME Type: {fileObj.file.type}</p>
            <p>Size: {fileObj.file.size} bytes</p> */}
            <hr />
          </div>
        ))}
      </div>
    );
  };
  const handleFileChange = (e) => {
    const newFile = e.target.files[0];
    setFiles([...files, { file: newFile, fileName }]);
    setFileName(''); // Clear fileName for the next file
    setChoosedFile('')
  };
  const handleFileNameChange = (e) => {
    setFileName(e.target.value);
  };
  const handleLeaveType = (e) => {
    setLeaveType(e.target.value)
  }
  return (
    <div className="leave-form-container container">
      <h2 className="text-center mb-4">Leave Request Form</h2>
        <div className="mb-3">
          <label htmlFor="reason" className="">Reason:</label>
          <textarea

            className="form-control"
            id="reason"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
          />
        </div>
        <div className="mb-3">
          <label className="form-label">Select Date:</label>
          <div className="form-check">
            <input
              type="radio"
              id="singleDate"
              className="form-check-input"
              checked={isSingleDate}
              onChange={() => setIsSingleDate(true)}
            />
            <label htmlFor="singleDate" className="">Single Date</label>
          </div>
          <div className="form-check">
            <input
              type="radio"
              id="dateRange"
              className="form-check-input"
              checked={!isSingleDate}
              onChange={() => setIsSingleDate(false)}
            />
            <label htmlFor="dateRange" className="">Date Range</label>
          </div>
        </div>
        {isSingleDate ? (
          <div className="mb-3">
            <label htmlFor="selectedDate" className="">Select Date:</label>
            <DatePicker
              id="selectedDate"
              selected={selectedDate}
              onChange={(date) => setSelectedDate(date)}
              className="form-control custom-datepicker"
            />
          </div>
        ) : (
          <div className="mb-3">
            <label className="">Date Range:</label>
            <div className="d-flex justify-content-between">
              <div className="me-2">
                <label htmlFor="startDate" className="">From:</label>
                <DatePicker
                  id="startDate"
                  selected={startDate}
                  onChange={(date) => setStartDate(date)}
                  dateFormat="dd.MM.yyyy"
                  className="form-control custom-datepicker"
                />
              </div>
              <div>
                <label htmlFor="endDate" className="">To:</label>
                <DatePicker
                  id="endDate"
                  selected={endDate}
                  onChange={(date) => setEndDate(date)}
                  dateFormat="dd.MM.yyyy"
                  className="form-control custom-datepicker"
                />
              </div>
            </div>
          </div>
        )}
        <select id="select-staff" value={leaveType} onChange={handleLeaveType}>
          <option value="">Choose Leave Type</option>
          <option value="Normal">Normal</option>
          <option value="ML">ML</option>
          <option value="PL">PL</option>
          <option value="OD">OD</option>
        </select>
        {(leaveType!='' && leaveType!='Normal') &&
          <div className='mb-3'>
          <h1>File Upload</h1>
          <form>
            <div>
              <label htmlFor="fileName">File Name:</label>
              <input type="text" id="fileName" name='fileInfo' value={fileName} onChange={handleFileNameChange} />
            </div>
            <div>
              <label htmlFor="file">Choose File:</label>
              <input type="file" id="file" name='files' value={choosedFile} onChange={handleFileChange} />
            </div>
          </form>

          {/* Display selected files information */}
          {files.length > 0 && renderFileList()}

        </div>
        }
        <button
          type="button"
          className="btn btn-success"
          onClick={handleSubmit}
        >
          Submit
        </button>
    </div>
  );
};

export default LeaveForm;

