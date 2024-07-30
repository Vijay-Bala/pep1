import React, { useEffect, useState } from 'react';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';

const CustomDatePicker = ({ onDateChange ,date}) => {
  const [selectedDate, setSelectedDate] = useState(null);

 
  const handleChange = (date) => {
    setSelectedDate(date);
    onDateChange(date);
  };
  return (
    <DatePicker
      selected={date!=null?selectedDate:''}
      onChange={handleChange}
      dateFormat="yyyy-MM-dd"
      placeholderText="Select a date"
    />
  );
};

export default CustomDatePicker;