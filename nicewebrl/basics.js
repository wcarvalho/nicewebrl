async function isFullscreen() {
  try {
    return document.fullscreenElement != null;
  } catch (error) {
    console.warn('Fullscreen check failed:', error);
    return false;  // Return a safe default
  }
} 

async function getImageSeenTime() {
  return window.imageSeenTime;
} 

// Function to send ping to server every 30 seconds
async function pingServer() {
  console.log('Starting ping loop');
  while (true) {
    try {
      const message = `Ping ${Math.floor(Math.random() * 10000)}`;
      // Await the asynchronous emitEvent call
      await emitEvent('ping', { message: message });
      console.log(`Ping sent: ${message}`);
    } catch (err) {
      console.error('Error pinging server:', err);
    }
    // Wait for 30 seconds before sending the next ping
    seconds = 30
    try {
      await new Promise(resolve => setTimeout(resolve, seconds * 1000));
    } catch (err) {
      console.error('Error in ping loop:', err);
    }
  }
}

// Function to toggle spacebar behavior
let spacebarPrevented = false; // Default to preventing spacebar
function preventDefaultSpacebarBehavior(shouldPrevent) {
  spacebarPrevented = shouldPrevent;
  return spacebarPrevented;
}

document.addEventListener('DOMContentLoaded', async function () {

  ////////////////
  // Start pinging the server once the DOM content is fully loaded
  ////////////////
  pingServer();

  ////////////////
  // remove default behavior
  ////////////////
  window.debug = 0;
  window.require_fullscreen = false;
  window.accept_keys = false;
  window.next_states = null;

  ////////////////
  // Prevent spacebar from toggling fullscreen
  ////////////////
  document.addEventListener('keydown', function(event) {
    // Skip if the chat input is focused
    if (document.activeElement && document.activeElement.id === 'chat-input') {
      return;
    }
    
    // Check if the key pressed is spacebar
    if ((event.key === " " || event.code === "Space") && spacebarPrevented) {
      // Prevent the default action (toggling fullscreen)
      event.preventDefault();
      console.log('prevented spacebar');
    }
  }, true); // Using capturing phase to catch the event before other handlers

  ////////////////
  // how to handle key presses?
  ////////////////
  document.addEventListener('keydown', async function (event) {
    // Skip if the chat input is focused
    if (document.activeElement && document.activeElement.id === 'chat-input') {
      return;
    }
    
    // Prevent default behavior for arrow keys
    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) {
      event.preventDefault();
    }

    // Handle key presses
    console.log(event.key);
    if (window.next_states !== null && window.accept_keys && event.key in window.next_states) {
      if (!window.require_fullscreen || await isFullscreen() ) {
        next_state = window.next_states[event.key];
        window.next_states = null;
        var imgElement = document.getElementById('stateImage');
        if (imgElement !== null) {
          imgElement.src = next_state;
        }
        window.next_imageSeenTime = new Date();
        console.log('set new image');
      }
      // Record the current time when the keydown event occurs
      var keydownTime = new Date();
      // Await the asynchronous emitEvent call
      await emitEvent('key_pressed', {
        key: event.key,
        keydownTime: keydownTime,
        imageSeenTime: window.imageSeenTime
      });
    }
  });
})