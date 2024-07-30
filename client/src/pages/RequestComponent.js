import React,{ useState,useEffect } from 'react'
import axios from 'axios'
import '../css/RequestComponent.css'

const RequestComponent = ({user,selectedStaff}) => {
    const [requests,setRequests] = useState([])
    const [requestDetails,setRequestDetails] = useState(null)
    const [individualRequest,setIndividualRequest] = useState(false)
    const [allRequest,setAllRequest] = useState(true)
    const [files,setFiles] = useState(null)
    const [messageByFaculty,setMessageByFaculty] = useState('NA')
    const [backButtonClicked,setBackButtonClicked]=useState(false)
    useEffect(() => {
      const fetchData = async () => {
        try {
          const response = await axios.post('http://localhost:4000/requestPage',{
            class:user.name
          });
  
          const d = response.data;
          console.log("receivved: ",d);
          if (d.status) {
            // If the response is successful, you can set the PDF data
            console.log("set");
            setRequests(d.requests);

          } else {
            // Handle errors
            console.error('Error:', d.message);
          }
        } catch (error) {
          console.error('Fetch error:', error.message);
        }
      };
  
      fetchData();
    }, [user,backButtonClicked]);


    const handleIndividualRequest = async (request) => {
      try{
        const response = await axios.post('http://localhost:4000/getDocsForRequestId',{
          id:(request._id).toString()
        })
        const d = response.data
        if(d.status){
          setFiles(d.files)
          console.log("file data: ",d.files);
        }else{
          setFiles(null)
        }
      }catch(err){
        console.log(err);
      }

      setRequestDetails(request)
      setIndividualRequest(true)
      setAllRequest(false)
    }
    const handleViewPdf = () => {
      // Create a Blob from the PDF buffer
      // console.log("handlepdf: ",requestDetails.conId);
      console.log("pdf clicked");
      if(requestDetails.pdfUrl){
        // console.log("TYPE: ",requestDetails.pdfUrl);
        const d = Uint8Array.from(requestDetails.pdfUrl.data);
        const blob = new Blob([d], { type: 'application/pdf' });
  
        // Create a URL for the Blob
        const url = URL.createObjectURL(blob);
  
        // Open a new tab with the PDF content
        window.open(url, '_blank');
      }
    }
    const handleBack = () => {
      setIndividualRequest(false)
      setAllRequest(true)
      setBackButtonClicked(true)
    }
    const handleStatus = async(status) => {
      if(selectedStaff===''){
        alert('choose staff')
        return;
      }
      console.log("status: ",status);
      console.log("role: ",user.role);
      console.log("staff: ",selectedStaff);
      const response = await axios.post("http://localhost:4000/handleStatus",{
        messageByFaculty:messageByFaculty,
        selectedStaff:selectedStaff,
        status:status,
        requestId:requestDetails._id,
        userId:requestDetails.userId,
        role:user.role
      })
      const res = response.data;
      console.log("handlestatus: ",res);
      if(res.status){
        setMessageByFaculty('')
        handleBack()
      }
    }
    const handleMessage = (e) => {
      setMessageByFaculty(e.target.value)
    }

    return (
      <div className="main-Request">
        {allRequest && (
          <div className="request-container">
            <h1 className="section-title">User Requests</h1>
            {user && <p className="user-name">Logged in as: {user.name}</p>}
            {requests && (
              <div className="request-list">
                {requests.map((request) => (
                  <div className="request-row" key={request._id}>
                    <div className="user-details">
                      <h1>{request && request.userDetail[0]}</h1>
                      <h1>{request && request.userDetail[1]}</h1>
                      <h1>{request && request.userDetail[2]}</h1>
                    </div>
                    <button className="request-click" onClick={() => handleIndividualRequest(request)}>Click</button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        {
          individualRequest && <div className="individual-Request">
            <h1>{requestDetails && requestDetails.userDetail[0]}</h1>
            <h1>{requestDetails && requestDetails.userDetail[1]}</h1>
            <h1>{requestDetails && requestDetails.userDetail[2]}</h1>
            <h1>{requestDetails && requestDetails.reason}</h1>
            <h1>{requestDetails && requestDetails.singleDate}</h1>
            <input type="text" value={messageByFaculty} onChange={handleMessage} placeholder="Message to Student"></input>
            <button onClick={handleViewPdf}>View PDF</button>
            <button onClick={handleBack}>Back</button>
            <button onClick={()=>handleStatus("accept")}>Accept</button>
            <button onClick={()=>handleStatus("reject")}>Reject</button>
            
          </div>
        }

        {
          individualRequest && files && <div>
            <h2>Data Display</h2>
            <ul>
              {files.map((item, index) => (
                <li key={index}>
                  <strong>Content Type:</strong> {item.contentType}<br />
                  {/* Display the content based on content type */}
                  <RenderContent item={item} index={index}/>
                </li>
              ))}
            </ul>
          </div>
        }
      </div>
    )


  }

  const RenderContent = ({ item, index }) => {
    const [base, setBase] = useState('');
  
    const RequestPdf = async(data) => {
      // Create a Blob from the PDF buffer
      // console.log("handlepdf: ",requestDetails.conId);
      console.log("request pdf clicked");
      if(data.data){
        // console.log("TYPE: ",requestDetails.pdfUrl);
        const d = Uint8Array.from(data.data);
        const blob = new Blob([d], { type: 'application/pdf' });
   
        // Create a URL for the Blob
        const url = URL.createObjectURL(blob);
  
        // Open a new tab with the PDF content
        window.open(url, '_blank');
      }
    }
    
  
    return (
      <div>
        {item.contentType.startsWith('image/') && (
          <img src={`data:${item.contentType};base64,${item.data.toString('base64')}`} alt={`Image ${index}`} />
        )}
        {item.contentType === 'application/pdf' && (
          <button onClick={() => RequestPdf(item.data)}>View PDF</button>
        )}
      </div>
    );
  };
  
export default RequestComponent