import React from 'react';

const LeaveTable = ({ info }) => {
  console.log("info: ", info);

  return (
    <div>
      <h2>Student Leave Information</h2>
      <table>
        <thead>
          <tr>
            <th>Regisno</th>
            <th>Rollno</th>
            <th>Name</th>
            {[...Array(10)].map((_, index) => (
              <th key={index}>Date {index + 1}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {info && Object.keys(info).map((studentName) => {
            const studentData = info[studentName];
            const dateCols = Array.from({ length: 10 }, (_, index) => (
              studentData.data[index] ? studentData.data[index].date : ''
            ));
            const arrDates = dateCols.map((date) => date).filter(date => date !== '');

            arrDates.sort();

            return (
              <tr key={studentName}>
                <td>{studentData.regisno}</td>
                <td>{studentData.rollno}</td>
                <td>{studentName}</td>
                {arrDates.map((date, index) => (
                  <td key={index} title={getReasonForDate(studentData.data, date)}>
                    {date}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

const getReasonForDate = (data, date) => {
  const entry = data.find((item) => item.date === date);
  let str = '';
  if(entry.leaveReason!='') str+=`Reason: ${entry.leaveReason}\n`;
  str+=`LeaveApplied: ${entry.leaveApplied}\nLeaveApproved: ${entry.leaveApproved}`
  return str!='' ? str : '';
};

export default LeaveTable;
