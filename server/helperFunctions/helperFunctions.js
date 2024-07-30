const { PDFDocument, rgb } = require('pdf-lib');
const fs = require('fs')
const path = require('path');
const {format} = require('date-fns');
const { User, Request, File, Student } = require("../model/authModel");

function helper64(d){
    const base64String = Buffer.from(d).toString('base64');
    const imageUrl = `data:${contentType};base64,${base64String}`;
    return imageUrl
}
function getDatesBetween(startDate, endDate) {
    const dates = [];
    let currentDate = new Date(startDate);
    endDate = new Date(endDate);
  
    while (currentDate <= endDate) {
      dates.push(new Date(currentDate));
      currentDate.setDate(currentDate.getDate() + 1);
    }
  
    return dates;
  }
  function convertDateFormat(inputDate) {
    const parts = inputDate.split('.');
    if (parts.length === 3) {
      const [day, month, year] = parts;
      const formattedDate = `${year}-${month}-${day}`;
      return formattedDate;
    } else {
      // Handle invalid input
      console.error('Invalid date format');
      return null;
    }
  }
  async function markLeaveApplied(name, depart, section, reason, date, from, to, ID, leaveType) {
    console.log(name, depart, section, reason, date, from, to, ID, leaveType);
    let datesToUpdate = [];
    if (date !== " ") {
        date.replace(/\./g,'-')
        console.log(convertDateFormat(date));
        datesToUpdate.push(convertDateFormat(date));
    } else {
        datesToUpdate.push(convertDateFormat(from));
        datesToUpdate.push(convertDateFormat(to));
    }
    datesToUpdate=datesToUpdate.map((date) => new Date(date));
    console.log(datesToUpdate);
    // const dateObjects = datesToUpdate.map((date) => new Date(`${date}T00:00:00.000Z`));
    try {
        const result = await Student.updateMany(
            {
              depart: depart,
              section: section,
              'students.name': name,
              date: { $in: datesToUpdate},
            },
            {
              $set: {
                'students.$.leaveApplied': true,
                'students.$.present': false,
                'students.$.leaveReason': reason,
                'students.$.leaveRequestId': ID,
                'students.$.leaveCategory': leaveType,
              },
            }
          );
          
  
        console.log(result);
        return true;
    } catch (err) {
        console.error("Error:", err);
        return false;
    }
  }

async function drawTextOnPDF(pdfBytes,user,inputStudent) {
    const textSize = 14; 
    const pdfDoc = await PDFDocument.load(pdfBytes);
    const page = pdfDoc.getPages()[0];
    
    const pngImageBytes = fs.readFileSync(path.join(__dirname,`../../signs/${user._id}.png`));

    const pngImage = await pdfDoc.embedPng(pngImageBytes);
    const pngDims = pngImage.scale(0.20,{ skipRescale: true });

        page.drawImage(pngImage, {
            x: 330, 
            y: 140, 
            width: pngDims.width,
            height: pngDims.height,
        });
        page.drawText(user.name, {
            x: 300,
            y: page.getHeight() - 252,
            size: textSize,
        });
        page.drawText(user.rollno, {
            x: 300,
            y: page.getHeight() - 275,
            size: textSize,
        });
        page.drawText("UG", {
            x: 300,
            y: page.getHeight() - 298,
            size: textSize,
        });
        page.drawText(user.depart, {
            x: 300,
            y: page.getHeight() - 322,
            size: textSize,
        });
        page.drawText("B.E", {
            x: 450,
            y: page.getHeight() - 298,
            size: textSize,
        });
        page.drawText(user.section, {
            x: 300,
            y: page.getHeight() - 345,
            size: textSize,
        });
        page.drawText(user.leaves.toString(), {
            x: 340,
            y: page.getHeight() - 365,
            size: textSize,
        });
        page.drawText(user.leaves.toString(), {
            x: 340,
            y: page.getHeight() - 443,
            size: textSize,
        });
        page.drawText(inputStudent.reason, {
            x: 230,
            y: page.getHeight() - 460,
            size: textSize,
        });
        if(inputStudent.startDate==" " && inputStudent.endDate==" "){
            page.drawText(inputStudent.selectedDate, {
                x: 242,
                y: page.getHeight() - 491,
                size: 13,
            });
        }else{
            page.drawText(inputStudent.startDate, {
                x: 250,
                y: 330,
                size: 13,
            });
            page.drawText(inputStudent.endDate, {
                x: 345,
                y: 330,
                size: 13,
            });
        }
        
        // page.drawText(inputStudent.councellor, {
        //     x: 105,
        //     y: page.getHeight() - 750,
        //     size: textSize,
        // });
        // page.drawText(user.hod, {
        //     x: 380,
        //     y: page.getHeight() - 750,
        //     size: textSize,
        // });

    // Save the modified PDF
    const modifiedPdfBytes = await pdfDoc.save();

    return Buffer.from(modifiedPdfBytes);
}

async function addSignature(pdfBytes,role,facultyName){
    console.log("councellor: "+`${facultyName}`);
    const textSize = 14; 
    const pdfDoc = await PDFDocument.load(pdfBytes);
    const page = pdfDoc.getPages()[0];
    
    const pngImageBytes = fs.readFileSync(path.join(__dirname,`../../signs/${facultyName}.png`));

    // Embed the PNG image in the PDF
    const pngImage = await pdfDoc.embedPng(pngImageBytes);
    const pngDims = pngImage.scale(0.20,{ skipRescale: true });
    console.log("role final : ",role);
    if(role==="councellor"){
        console.log("hello");
        page.drawImage(pngImage, {
            x: 105, 
            y: 50, 
            width: pngDims.width,
            height: pngDims.height,
        });
        page.drawText(facultyName, {
            x: 105,
            y: page.getHeight() - 750,
            size: textSize,
        });
    }
    else if(role==="hod"){
        page.drawImage(pngImage, {
            x: 370, 
            y: 50, 
            width: pngDims.width,
            height: pngDims.height,
        });
        page.drawText(facultyName, {
            x: 380,
            y: page.getHeight() - 750,
            size: textSize,
        });
    }
    const modifiedPdfBytes = await pdfDoc.save();

    return Buffer.from(modifiedPdfBytes);
}

module.exports = {
    drawTextOnPDF,
    addSignature,
    markLeaveApplied,
    helper64,
    convertDateFormat,
    getDatesBetween
}