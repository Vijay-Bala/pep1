import React, { useState, useEffect } from 'react';
import axios from 'axios';

const HodDepartmentDetails = ({ user, domains }) => {
  const [data, setData] = useState([]);
  const [domain, setDomain] = useState('');
  const [batch, setBatch] = useState('');
  const [loading, setLoading] = useState(true); // State to track loading state
  const [error, setError] = useState(null); // State to track errors

  useEffect(() => {
    const fetchDetails = async () => {
      setLoading(true); // Set loading state when fetching data
      setError(null); // Clear any previous errors

      try {
        const response = await axios.post("http://localhost:4000/pepDetails", {
          domains: domains
        });
        if (response.data.status === true) {
          setData(response.data.result);
        } else {
          setError("No data available."); // Set custom error message if no data is available
        }
      } catch (err) {
        console.error(err);
        setError("Failed to fetch data."); // Set error message for fetch failure
      } finally {
        setLoading(false); // Always set loading to false when request completes
      }
    };
    fetchDetails();
  }, [domains]); // Fetch data whenever domains prop changes

  const handleFilter = async () => {
    setLoading(true); // Set loading state when filtering
    setError(null); // Clear any previous errors

    try {
      const response = await axios.post("http://localhost:4000/filterDetails", {
        domain,
        batch: parseInt(batch, 10)
      });
      if (response.data.status === true) {
        setData(response.data.result);
      } else {
        setData([]); // Clear data if no results found
        setError("No matching results found."); // Set custom error message for no results
      }
    } catch (err) {
      console.error(err);
      setError("Failed to filter data."); // Set error message for filter failure
    } finally {
      setLoading(false); // Always set loading to false when request completes
    }
  };

  return (
    <div>
      <h2>Student Leave Information</h2>
      <select value={domain} onChange={(e) => setDomain(e.target.value)}>
        <option value="">Choose Domain</option>
        {domains.map((d, index) => (
          <option key={index} value={d}>{d.toUpperCase()}</option>
        ))}
      </select>
      <select value={batch} onChange={(e) => setBatch(e.target.value)}>
        <option value="">Choose Batch</option>
        <option value="2021">2021</option>
        <option value="2022">2022</option>
        {/* Add other batches as needed */}
      </select>
      <button onClick={handleFilter}>Filter</button>
      <br />
      <br />
      {loading && <p>Loading...</p>}
      {error && <p>{error}</p>}
      
      {data.length > 0 && <DepartLeaveTable information={data} />}
      {/* Display table only if there is data */}
    </div>
  );
};

const DepartLeaveTable = ({ information }) => {
  if (information.length === 0) return <div>No data available</div>;

  // Process data to generate the table
  const map = {};
  information.forEach(eachDayData => {
    eachDayData.students.forEach(student => {
      if (!map[eachDayData.date]) {
        map[eachDayData.date] = {};
      }
      if (!map[eachDayData.date][student.regisno]) {
        map[eachDayData.date][student.regisno] = {};
      }
      map[eachDayData.date][student.regisno]["forePresent"] = student.forePresent;
      map[eachDayData.date][student.regisno]["afterPresent"] = student.afterPresent;
    });
  });

  // Get sorted date columns
  const dates = Object.keys(map).sort();

  return (
    <div style={{ overflowX: 'auto' }}>
      <table>
        <thead>
          <tr>
            <th style={{ whiteSpace: "nowrap" }}>Name</th>
            <th>Reg No</th>
            <th>Department</th>
            {dates.map((date, index) => (
              <th key={index}>{new Date(date).toLocaleDateString()}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {information[0].students.map((entry, index) => (
            <tr key={index}>
              <td style={{ whiteSpace: "nowrap" }}>{entry.name}</td>
              <td>{entry.regisno}</td>
              <td>{entry.dept}</td>
              {dates.map(date => (
                <td key={date}>
                  {map[date][entry.regisno]["forePresent"] ? "P" : "A"}
                  {" | "}
                  {map[date][entry.regisno]["afterPresent"] ? "P" : "A"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default HodDepartmentDetails;
