const mongoose = require('mongoose');
const { User, Request, File } = require("./model/authModel");

const createNewDayDocument = async () => {
  console.log("inside createNewDayDocument");
  try {
    for(let i=0;i<10;i++){
        console.log(i);
    }
    // const newDocument = await User.create({
    //   name: "ECE A",
    //   role: "councellor",
    //   regino: 66666,
    //   password: "66666",
    //   staffs: ["ABCD", "EFGH"],
    //   batch: 2021
    // });

    // console.log(`New document with id ${newDocument._id}`);
  } catch (error) {
    console.error('Error creating new day document:', error);
  } finally {
    console.log("final");
    console.log("final");
  }
};

createNewDayDocument();

