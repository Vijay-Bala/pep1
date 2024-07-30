import React from 'react';
import '../css/DetailsComponent.css'

const DetailsComponent = ({ user }) => {
  return (
    <div className="details-container">
      <h1 className="detail-heading">Profile</h1>
      <div className="detail-item">
        <strong>Role:</strong> {user && user.role}
      </div>
      <div className="detail-item">
        <strong>Domain:</strong> {user && user.domain}
      </div>
    </div>
  );
};
export default DetailsComponent;