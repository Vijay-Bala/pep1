const mongoose = require("mongoose");


const userSchema= new mongoose.Schema({
  domain: String, 
  role: String,
  password: String,
});

const StudentSchema = new mongoose.Schema({
  name:String,
  dept:String,
  batch:Number,
  regisno:Number,
  rollno:String,
  afterPresent:Boolean,
  forePresent:Boolean,
  afterDetail:String,
  foreDetail:String
})
const MainSchema = new mongoose.Schema({
  date:{
      type:Date
  },
  domain:{
      type:String,
  },
  batch:{
    type:Number
  },
  students:[StudentSchema]
})
const SeparateStudentSchema = new mongoose.Schema({
  name:String,
  dept:String,
  batch:Number,
  regisno:Number,
  rollno:String,
  sessionsAbsentCount:Number,
  domain:String,
  noOfProjects:Number
})
const StudentMainSchema = new mongoose.Schema({
  name:String,
  dept:String,
  batch:Number,
  regisno:Number,
  rollno:String,
  domain:String
})

const User = mongoose.model("users", userSchema);
const Main = mongoose.model("mains", MainSchema);
const Student = mongoose.model("students", SeparateStudentSchema);
const IndividualStudent = mongoose.model("individual", StudentMainSchema);

module.exports = {
  User, Main, Student, IndividualStudent
};
