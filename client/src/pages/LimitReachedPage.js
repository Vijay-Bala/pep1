import React from 'react';

const LimitReachedPage = () => {
  return (
    <div style={styles.limitContainer}>
      <h1 style={styles.limitHeading}>Leave Limit Reached</h1>
      <p style={styles.limitMessage}>
        Sorry, you have reached your limit for this action. Please contact support for assistance.
      </p>
      {/* You can add additional styling or UI elements as needed */}
    </div>
  );
};

const styles = {
  limitContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100vh',
    background: '#f0f0f0', // Example background color
  },
  limitHeading: {
    fontSize: '2rem',
    color: '#333', // Example text color
  },
  limitMessage: {
    fontSize: '1rem',
    color: '#555', // Example text color
    textAlign: 'center',
    maxWidth: '400px', // Example maximum width for the message
    marginTop: '20px',
  },
};

export default LimitReachedPage;
