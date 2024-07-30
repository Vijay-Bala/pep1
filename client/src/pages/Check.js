import React, { useEffect, useState } from 'react';
import axios from 'axios';

const Check = () => {
  const [pdfBuffer, setPdfBuffer] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.get("http://localhost:4000/pdf");
        const pdfData = res.data.user;
        setPdfBuffer(pdfData);
      } catch (err) {
        console.error(err);
      }
    };

    fetchData();
  }, []);



  return (
    <div>
      <button >Download PDF</button>
    </div>
  );
};

export default Check;
