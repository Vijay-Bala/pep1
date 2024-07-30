import React, { useState } from 'react';
import axios from 'axios'

const FileUploader = () => {
  const [files, setFiles] = useState([]);
  const [fileName, setFileName] = useState('');
  const [choosedFile, setChoosedFile] = useState('');

  const handleFileChange = (e) => {
    const newFile = e.target.files[0];
    setFiles([...files, { file: newFile, fileName }]);
    setFileName(''); // Clear fileName for the next file
    setChoosedFile('')
  };

  const handleFileNameChange = (e) => {
    setFileName(e.target.value);
  };
  const handleUpload = async () => {
    try {
      const formData = new FormData();
      files.forEach((fileObj, index) => {
        formData.append(`files`, fileObj.file);
        formData.append(`fileName`, fileObj.fileName);
      });
  
      // Make the POST request to upload files
      await axios.post('http://localhost:4000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
  
      // Clear the files state for the next upload
      setFiles([]);
  
      alert('Files uploaded successfully!');
    } catch (error) {
      console.error('Error uploading files:', error);
      alert('Error uploading files');
    }
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

  return (
    <div>
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

      <button onClick={handleUpload}>Upload File</button>
    </div>
  );
};

export default FileUploader;


