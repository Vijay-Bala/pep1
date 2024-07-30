import React,{ useState,useEffect } from 'react'
import axios from 'axios'

const HodRequestComponent = ({user,selectedStaff}) => {
    const [requests,setRequests] = useState([])
    const [categoryRequests,setCategoryRequests] = useState([])
    const [requestDetails,setRequestDetails] = useState(null)
    const [individualRequest,setIndividualRequest] = useState(false)
    const [allRequest,setAllRequest] = useState(true)
    const [files,setFiles] = useState(null)
    const [leaveCategory,setLeaveCategory] = useState('')
    const [backButtonClicked,setBackButtonClicked]=useState(false)
    const [selectedRequests, setSelectedRequests] = useState([]);
    const [messageByFaculty,setMessageByFaculty] = useState('NA')
    const [isChecked,setIsChecked] = useState(false)
    const [refreshKey, setRefreshKey] = useState(0);

    useEffect(() => {
      const fetchData = async () => {
        try {
          const response = await axios.post('http://localhost:4000/hodrequestPage',{
            depart:user.name
          });
  
          const d = response.data;
          console.log("Requests: ",d);
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
    }, [user,refreshKey]);

    useEffect(()=>{
      console.log("r:  ",requests);
    },[individualRequest])

    useEffect(()=>{
      let arr=requests.filter((request) => request.leaveType===leaveCategory)
      setCategoryRequests(arr)
    },[requests,leaveCategory])


    const handleIndividualRequest = async (request) => {
      setIsChecked(false)
      setSelectedRequests([])
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



    const handleStatus = async(status) => {
      if(selectedStaff===''){
        alert('choose staff')
        return;
      }
      const response = await axios.post("http://localhost:4000/handleStatus",{
        messageByFaculty:messageByFaculty,
        selectedStaff:selectedStaff,
        status:status,
        requestId:requestDetails._id,
        userId:requestDetails.userId,
        role:"hod"
      })
      const res = response.data;
      console.log("handlestatus: ",res);
      if(res.status){
        handleBack()
      }
    }
    const handleMessage = (e) => {
        setMessageByFaculty(e.target.value)
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
        setRefreshKey((prevKey) => prevKey + 1);
    }

    // select feature code


    const handleIndividualRequestClick = (request) => {
      const isSelected = selectedRequests.includes(request._id);
      setSelectedRequests((prevSelected) =>
        isSelected ? prevSelected.filter((id) => id !== request._id) : [...prevSelected, request._id]
      );
    };

    const handleCheckboxChange = (request) => {
      handleIndividualRequestClick(request);
    };

    const handleSelectAllClick = (e) => {
      setIsChecked((prev)=>!prev)
        if(e.target.checked===true){
          const allRequestIds = categoryRequests.map((request) => request._id);
          setSelectedRequests(allRequestIds);
        }else{  
          setSelectedRequests([]);
        }
    };


    const handleAcceptAllClick = async () => {
      if(selectedStaff===''){
        alert('choose staff')
        return;
      }
      if(selectedRequests.length===0){
        alert('choose request')
        return;
      }
      const response1 = await axios.post('http://localhost:4000/handleAcceptAll', {
        selectedRequests:selectedRequests,
        role:"hod",
        selectedStaff:selectedStaff
      })
      const response2 = await axios.post('http://localhost:4000/handleStudentSchema', {
        selectedRequests:selectedRequests,
        role:"hod",
        selectedStaff:selectedStaff
      })
      const d = response2.data;
      console.log(d);
      if(d.status===true){
        
      }
    }


    useEffect(()=>{
      console.log("leave: ",leaveCategory);
        setIsChecked(false)
        setSelectedRequests([])
    },[leaveCategory])


    const handleLeaveType = (e) => {
      console.log("handle");
      const le = e.target.value
      setLeaveCategory(le)
    }

    


    return (
      <div className="main-Request">
        {allRequest && (
          <div className="request-container">
            <h1 className="section-title">User Requests</h1>
            {user && <p className="user-name">Logged in as: {user.name}</p>}
            <select value={leaveCategory} onChange={handleLeaveType}>
              <option value="">Choose Leave Type</option>
              <option value="Normal">Normal</option>
              <option value="ML">ML</option>
              <option value="PL">PL</option>
              <option value="OD">OD</option>
            </select>
          {categoryRequests && (
              <div className='leave-container'>
                <div className="request-list">
                  {leaveCategory && (categoryRequests.length>0 ) && <div className='leaveCategory'>
                    <span>Select All</span>
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={handleSelectAllClick}
                    />
                  </div>}
                  {categoryRequests.map((request) => (
                    <div>
                      <div className="request-row" key={request._id}>
                        <div className="user-details">
                          <h1>{request && request.userDetail[0]}</h1>
                          <h1>{request && request.userDetail[1]}</h1>
                          <h1>{request && request.userDetail[2]}</h1>
                        </div>
                        <input
                          type="checkbox"
                          checked={selectedRequests.includes(request._id)}
                          onChange={() => handleCheckboxChange(request)}
                        />
                        <button className="request-click" onClick={() => handleIndividualRequest(request)}>Click</button>
                      </div>
                    </div>
                  )
                  )}
                </div>
                {leaveCategory!=''&& (categoryRequests.length>0 ) && <button onClick={handleAcceptAllClick}>Accept All</button>}
              </div>
            )}  
          </div>
        )}
        {
          individualRequest &&  <div className="individual-Request">
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

  const RenderContent =({item,index}) => {
    const [base,setBase] = useState('')

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
    useEffect(()=>{
      const handleImage = async () => {
        try{  
          console.log("d: ",item.data);
          const response = await axios.post('http://localhost:4000/convertTobase64',{
            varun:"saii",
            data:item.data.data
          })
          const d = response.data.image.data
          console.log("f: ",d);
          setBase(d)
        }catch(err){
          console.log(err);
        }     
      }
      handleImage()
    },[item])
    
    return (
      <div>
        {(item.contentType.startsWith('image/jpeg')) &&
        <img src={`data:image/png;base64,${base}`} alt={`Image ${index}`} />
        }
        {
          (item.contentType === 'application/pdf') && 
          <button onClick={()=>RequestPdf(item.data)}>View</button>
        }
      </div>
    )
  };  
export default HodRequestComponent