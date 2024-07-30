import React from "react";
import { Routes, Route, BrowserRouter } from "react-router-dom";
import Login from "./pages/Login";
import Cards from "./pages/Cards";
import "react-toastify/dist/ReactToastify.css";
import StudentHome from "./pages/StudentHome";
import Check from "./pages/Check";
import Select from "./pages/Select";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route exact path="/login" element={<Login />} />
        <Route exact path="/" element={<StudentHome />} />
        <Route exact path="/home" element={<StudentHome />} />
        <Route exact path="/check" element={<Check />} />
        <Route exact path="/select" element={<Select />} />

      </Routes>
    </BrowserRouter>
  );
}
